"""
Motor imagery BCI calibration: collect EEG trials, preprocess, fine-tune DeepConvNet.

Usage:
  python ml_model/MI_calibration.py
  python ml_model/MI_calibration.py --n-move 15 --n-idle 15 --unfreeze-blocks 1 --train-epochs 30 --lr 5e-4
  python ml_model/MI_calibration.py --skip-collection --fif-path data_collection/calibration_fifs/xxx.fif
"""

import os
import sys
import time
import random
import argparse
import warnings
from datetime import datetime

import pygame

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import mne

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

from mne_lsl.stream import StreamLSL

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from ml_model import models
from processing.preprocess_pipeline import MNEPipeline, epoch, epoch_filter, extract_tensor

# ── Constants ─────────────────────────────────────────────────────────────────

CH_NAMES = [
    "EEG LE-Pz", "EEG F4-Pz", "EEG C4-Pz", "EEG P4-Pz",
    "EEG P3-Pz", "EEG C3-Pz", "EEG F3-Pz", "Pz",
]
EXCLUDE      = {"Trigger", "Event"}
STREAM_NAME  = "WS-default"
SFREQ        = 300
TRIAL_DUR    = 4.0   # seconds pulled from stream per trial (full cue window)
EPOCH_TMAX   = 3.0   # tmax for MNE epoching; extra 1s is FIR filter edge buffer
ITI_MIN      = 4.0   # min random inter-trial interval (seconds) — idle epoch pulled here
ITI_MAX      = 7.0   # max random inter-trial interval

MODEL_PATH     = os.path.join(ROOT, "src", "models", "DeepConvNet_per_epoch.pt")
FIF_DIR        = os.path.join(ROOT, "data_collection", "calibration_fifs")
CALIBRATED_DIR = os.path.join(ROOT, "ml_model", "calibrated_models")

LABEL_MAP = {"idle": 0, "move": 1}

# Ordered from output back to input — used for progressive unfreezing
UNFREEZE_GROUPS = [
    "classifier",
    "block4_conv",
    "block3_conv",
    "block2_conv",
    "spatial_conv",
    "temporal_conv",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="EEG motor imagery calibration")
    p.add_argument("--n-move",          type=int,   default=10,
                   help="Number of MOVE trials (default: 10); idle count matches automatically")
    p.add_argument("--unfreeze-blocks", type=int,   default=0, choices=range(5),
                   help="Blocks to unfreeze: 0=classifier only … 4=all (default: 0)")
    p.add_argument("--train-epochs",    type=int,   default=20,
                   help="Fine-tuning epochs (default: 20)")
    p.add_argument("--lr",              type=float, default=1e-3,
                   help="Learning rate (default: 1e-3)")
    p.add_argument("--skip-collection", action="store_true",
                   help="Skip data collection; use --fif-path instead")
    p.add_argument("--fif-path",        type=str,   default=None,
                   help="Path to existing .fif file (requires --skip-collection)")
    p.add_argument("--mock-stream",     action="store_true",
                   help="Run paradigm with fake EEG data — no LSL connection needed")
    p.add_argument("--fullscreen",      action="store_true",
                   help="Run pygame window in fullscreen mode")
    return p.parse_args()


# ── Pygame display ────────────────────────────────────────────────────────────

_BG     = (20,  20,  20)
_WHITE  = (255, 255, 255)
_YELLOW = (255, 220,  50)
_CYAN   = (100, 200, 255)
_GRAY   = (140, 140, 140)


def _render(screen, lines: list):
    """Draw centered text lines. Each entry: (text, size, color, y_offset)."""
    screen.fill(_BG)
    cx = screen.get_width()  // 2
    cy = screen.get_height() // 2
    for text, size, color, y_off in lines:
        font = pygame.font.Font(None, size)
        surf = font.render(text, True, color)
        screen.blit(surf, surf.get_rect(center=(cx, cy + y_off)))
    pygame.display.flip()


def _pump(raise_on_quit: bool = True):
    """Process pending pygame events; raise KeyboardInterrupt on Q/Escape/close."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise KeyboardInterrupt
        if event.type == pygame.KEYDOWN and raise_on_quit:
            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                raise KeyboardInterrupt


def _sleep_events(seconds: float):
    """Sleep for `seconds` while keeping the event queue drained."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        _pump()
        pygame.time.wait(20)


def _wait_for_space(screen):
    """Show a 'Press SPACE' prompt and block until SPACE is pressed."""
    _render(screen, [
        ("Press SPACE to continue", 36, _GRAY, 160),
        ("(Q / Esc to abort)",      28, _GRAY, 200),
    ])
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    raise KeyboardInterrupt
        pygame.time.wait(20)


# ── LSL ───────────────────────────────────────────────────────────────────────

def attempt_lsl_connection(max_retries: int = 10, retry_delay: int = 2):
    for attempt in range(1, max_retries + 1):
        try:
            stream = StreamLSL(bufsize=TRIAL_DUR + 2.0, name=STREAM_NAME).connect()
            print(f"[calibration] Connected to LSL stream on attempt {attempt}.")
            return stream
        except Exception as e:
            print(f"[calibration] Attempt {attempt}/{max_retries} failed: {e}. "
                  f"Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
    return None


# ── Phase 1: Data Collection ──────────────────────────────────────────────────

def run_collection(args) -> "str | None":
    """Collect cued EEG trials via LSL, assemble a FIF with annotations."""
    if args.mock_stream:
        stream    = None
        sfreq     = float(SFREQ)
        eeg_picks = CH_NAMES
        print("[calibration] MOCK MODE — no LSL connection, fake data will be used\n")
    else:
        stream = attempt_lsl_connection()
        if stream is None:
            print("[calibration] Could not connect — is dsi2lsl running?")
            return None

        all_ch    = stream.info["ch_names"]
        sfreq     = stream.info["sfreq"]
        eeg_picks = [ch for ch in all_ch if ch not in EXCLUDE]

        if eeg_picks != CH_NAMES:
            stream.disconnect()
            raise RuntimeError(
                f"Channel order mismatch.\n  Expected: {CH_NAMES}\n  Got:      {eeg_picks}"
            )

    n_total    = args.n_move
    trial_data = []   # list of (label, np.ndarray shape (8, n_samples))

    pygame.init()
    flags  = pygame.FULLSCREEN if args.fullscreen else 0
    screen = pygame.display.set_mode((900, 600), flags)
    pygame.display.set_caption("MI Calibration")
    pygame.mouse.set_visible(False)

    print(f"[calibration] Starting: {n_total} move trials (idle collected from ITIs)")

    try:
        for i in range(n_total):
            counter = f"Trial {i+1} / {n_total}"

            # Fixation cross
            _render(screen, [
                (counter, 28, _GRAY,  -220),
                ("+",    120, _WHITE,     0),
            ])
            _sleep_events(4.0)

            # Ready
            _render(screen, [
                (counter, 28, _GRAY,  -220),
                ("READY", 80, _WHITE,     0),
            ])
            _sleep_events(1.5)

            # Cue — EEG window
            _render(screen, [
                (counter,          28, _GRAY,    -220),
                ("IMAGINE MOVING", 88, _WHITE,      0),
            ])
            _sleep_events(TRIAL_DUR)

            # Pull move epoch immediately after cue ends
            if stream is not None:
                data, _ = stream.get_data(winsize=TRIAL_DUR, picks=eeg_picks)
            else:
                data = np.random.randn(len(eeg_picks), int(TRIAL_DUR * sfreq))
            trial_data.append(("move", data.copy()))

            # Random ITI — blank screen, silently collect idle epoch at the end
            iti = random.uniform(ITI_MIN, ITI_MAX)
            _render(screen, [])   # blank
            _sleep_events(iti)

            if stream is not None:
                idle_data, _ = stream.get_data(winsize=TRIAL_DUR, picks=eeg_picks)
            else:
                idle_data = np.random.randn(len(eeg_picks), int(TRIAL_DUR * sfreq))
            trial_data.append(("idle", idle_data.copy()))

            # Press space to start next trial (skip prompt on last trial)
            if i < n_total - 1:
                _wait_for_space(screen)

    except KeyboardInterrupt:
        print(f"\n[calibration] Aborted after {sum(1 for l,_ in trial_data if l=='move')} move trials.")

    finally:
        if stream is not None:
            stream.disconnect()
        pygame.mouse.set_visible(True)
        pygame.quit()

    if len(trial_data) < 4:
        print(f"[calibration] Only {len(trial_data)} trials collected — need at least 4. Aborting.")
        return None

    # Assemble FIF
    os.makedirs(FIF_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fif_path  = os.path.join(FIF_DIR, f"{timestamp}_calibration.fif")

    all_data = np.concatenate([d for _, d in trial_data], axis=1)  # (8, N*n_samples)
    info     = mne.create_info(ch_names=eeg_picks, sfreq=float(sfreq), ch_types="eeg")
    raw      = mne.io.RawArray(all_data, info, verbose=False)

    n_trials    = len(trial_data)
    trial_dur_s = all_data.shape[1] / float(sfreq) / n_trials   # actual duration per trial
    onsets      = [i * trial_dur_s for i in range(n_trials)]
    durations   = [EPOCH_TMAX] * n_trials   # 3s annotation; 4th second is FIR edge buffer
    descriptions = [label for label, _ in trial_data]

    raw.set_annotations(mne.Annotations(
        onset=onsets, duration=durations, description=descriptions
    ))
    raw.save(fif_path, overwrite=True)

    counts = {lbl: descriptions.count(lbl) for lbl in set(descriptions)}
    print(f"[Phase 1] Saved {n_trials} trials {counts} → {fif_path}")
    return fif_path


# ── Phase 2: Preprocessing ────────────────────────────────────────────────────

def run_preprocessing(fif_path: str):
    """Preprocess FIF with the same per-epoch pipeline used during training."""
    print("\n[Phase 2] Preprocessing...")

    pipeline = MNEPipeline(fif_path)
    pipeline.add(epoch(tmin=0, tmax=EPOCH_TMAX))
    pipeline.add(epoch_filter(bands=[(13, 30)], notch_freq=60, ref="average"))
    raw = pipeline.run()

    X, y_raw = extract_tensor(scale=True)(raw)

    auto_ids = raw._event_id   # e.g. {'idle': 1, 'move': 2}
    remap = {v: LABEL_MAP[k] for k, v in auto_ids.items() if k in LABEL_MAP}
    y = np.array([remap[yi] for yi in y_raw])

    print(f"[Phase 2] X: {X.shape}  classes: {np.unique(y)}  counts: {np.bincount(y)}")

    for cls_idx, cls_name in [(0, "idle"), (1, "move")]:
        count = int((y == cls_idx).sum())
        if count < 2:
            raise ValueError(
                f"[Phase 2] Need >= 2 '{cls_name}' trials for fine-tuning, got {count}. "
                f"Re-run collection with more trials."
            )

    return X, y


# ── Phase 3: Fine-Tuning ──────────────────────────────────────────────────────

def apply_freeze_strategy(model: nn.Module, n_unfreeze: int) -> set:
    """Freeze all params; unfreeze the last (n_unfreeze+1) layer groups."""
    for p in model.parameters():
        p.requires_grad_(False)

    trainable = set(
        UNFREEZE_GROUPS if n_unfreeze >= 4 else UNFREEZE_GROUPS[:n_unfreeze + 1]
    )

    for name, p in model.named_parameters():
        top = name.split(".")[0]
        if top in trainable:
            p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[Phase 3] Trainable groups: {trainable}")
    print(f"[Phase 3] Trainable params: {n_trainable:,} / {n_total:,}")
    return trainable


def run_finetuning(X: np.ndarray, y: np.ndarray, args) -> str:
    """Load DeepConvNet, freeze strategy, fine-tune, save calibrated model."""
    print("\n[Phase 3] Fine-tuning...")

    model = models.DeepConvNet(n_channels=8, n_classes=2)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )

    trainable = apply_freeze_strategy(model, args.unfreeze_blocks)

    # Keep frozen BN layers in eval mode so their running stats aren't corrupted
    # by the small calibration batch.
    model.eval()
    for name, m in model.named_modules():
        top = name.split(".")[0] if "." in name else name
        if top in trainable:
            m.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    criterion = nn.NLLLoss()

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=8, shuffle=True)

    print(f"[Phase 3] Training {args.train_epochs} epochs on {len(X_t)} trials  lr={args.lr}")

    for ep_idx in range(args.train_epochs):
        ep_loss = 0.0
        correct = 0

        for X_b, y_b in loader:
            optimizer.zero_grad()
            out     = model(X_b)
            loss    = criterion(torch.log(out + 1e-10), y_b)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(X_b)
            correct += (out.argmax(1) == y_b).sum().item()

        if (ep_idx + 1) % 5 == 0 or ep_idx == 0:
            avg_loss = ep_loss / len(X_t)
            acc      = correct / len(X_t)
            print(f"  Epoch {ep_idx+1:3d}/{args.train_epochs}  loss={avg_loss:.4f}  acc={acc:.3f}")

    os.makedirs(CALIBRATED_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(CALIBRATED_DIR, f"{timestamp}_calibrated.pt")
    torch.save(model.state_dict(), save_path)
    print(f"[Phase 3] Saved calibrated model → {save_path}")
    return save_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  EEG Motor Imagery Calibration")
    print(f"  Trials:          {args.n_move} move + {args.n_move} idle (from ITIs)")
    print(f"  Unfreeze blocks: {args.unfreeze_blocks}  "
          f"({'classifier only' if args.unfreeze_blocks == 0 else 'see UNFREEZE_GROUPS'})")
    print(f"  Train epochs:    {args.train_epochs}  lr={args.lr}")
    print("=" * 60)

    # Phase 1
    if args.skip_collection:
        if args.fif_path is None:
            raise ValueError("--fif-path is required when using --skip-collection")
        fif_path = args.fif_path
        print(f"[Phase 1] Skipped — using {fif_path}")
    else:
        fif_path = run_collection(args)
        if fif_path is None:
            print("[main] Collection failed — exiting.")
            return

    # Phase 2
    X, y = run_preprocessing(fif_path)

    # Phase 3
    save_path = run_finetuning(X, y, args)

    print("\n[main] Calibration complete.")
    print(f"[main] Calibrated model: {save_path}")


if __name__ == "__main__":
    main()
