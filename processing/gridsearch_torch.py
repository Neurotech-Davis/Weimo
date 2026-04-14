# Import helpers
import sys
import os
sys.path.append(os.path.join(os.path.dirname('.'), '..'))

# Preprocessing
from preprocess_pipeline import (
    MNEPipeline, notch, bandpass, rereference,
    add_idle_class, add_fake_jaw_clench, epoch, extract_tensor,
)
import numpy as np
import json
import itertools
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold

# Importing models
from ml_model.models import (
    DeepConvNet, ShallowConvNet, EEGNet, ATCNet, Conformer, CTNet, MBMANet, PBT, LMDANet, EEGITNet
)

import torch
import torch.nn as nn

# =============================================================================
# Configurations
# =============================================================================

N_CHANNELS   = 8    # Number of EEG channels in your data
N_CLASSES    = 3    # idle, move, jaw_clench
SFREQ        = 300  # Sample rate
TRIAL_DUR    = 3.0  # seconds (tmax - tmin in epoch())
N_TIMEPOINTS = int(SFREQ * TRIAL_DUR)

FILES = [
    '../data_collection/annotated_eeg/chengyi_4_8_0.fif',
    '../data_collection/annotated_eeg/chengyi_4_8_1.fif',
    '../data_collection/annotated_eeg/chengyi_4_9_0.fif',
    '../data_collection/annotated_eeg/chengyi_4_9_1.fif',
    '../data_collection/annotated_eeg/chengyi_4_9_2.fif',
]
LABEL_MAP = {'idle': 0, 'move': 1, 'jaw_clench': 2}
MAP_LABEL = {v: k for k, v in LABEL_MAP.items()}

BANDPASS_CONFIGS = [
    [(8, 13)],
    [(13, 30)],
    [(8, 13), (13, 30)],
    [(4, 8), (8, 13), (13, 30)],
]

EPOCHS     = 100
BATCH_SIZE = 32
LR         = 1e-3
CV_FOLDS   = 5
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}  |  N_CHANNELS: {N_CHANNELS}  |  N_TIMEPOINTS: {N_TIMEPOINTS}")

# =============================================================================
# Load data
# =============================================================================

def load_tensor_data(files, bands, label_map):
    all_X, all_y = [], []
    for file in files:
        pipeline = MNEPipeline(file)
        pipeline.add(notch(60))
        pipeline.add(bandpass(bands))
        pipeline.add(rereference())
        pipeline.add(add_idle_class(window_dur=3.0, idle_start_min=1.0))
        pipeline.add(add_fake_jaw_clench())
        pipeline.add(epoch(tmin=0, tmax=TRIAL_DUR))

        raw = pipeline.run()
        X, y = extract_tensor(scale=True)(raw)   # (N, 1, C, T)

        auto_ids = raw._event_id
        remap = {v: label_map[k] for k, v in auto_ids.items() if k in label_map}
        y = np.array([remap[yi] for yi in y])

        T = X.shape[-1]
        if T > N_TIMEPOINTS:
            X = X[..., :N_TIMEPOINTS]
        elif T < N_TIMEPOINTS:
            pad = np.zeros((*X.shape[:-1], N_TIMEPOINTS - T), dtype=np.float32)
            X = np.concatenate([X, pad], axis=-1)

        all_X.append(X)
        all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"  Loaded: X={X.shape}, classes={np.unique(y, return_counts=True)}")
    return X, y

X, y = load_tensor_data(FILES, BANDPASS_CONFIGS[0], LABEL_MAP)

# =============================================================================
# Training utilities
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct   += (out.argmax(1) == yb).sum().item()
        n         += len(yb)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        total_loss += loss.item() * len(yb)
        correct   += (out.argmax(1) == yb).sum().item()
        n         += len(yb)
    return total_loss / n, correct / n


def run_cv(model_fn, X, y, config):
    """
    model_fn : callable with no args → returns a fresh model
    config   : dict with keys epochs, batch_size, lr, cv_folds, device, patience
    """
    device    = config['device']
    criterion = nn.CrossEntropyLoss()
    skf       = StratifiedKFold(n_splits=config['cv_folds'], shuffle=True, random_state=42)

    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(Xt, yt)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n── Fold {fold+1}/{config['cv_folds']} ──")

        train_loader = DataLoader(Subset(dataset, train_idx),
                                  batch_size=config['batch_size'], shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx),
                                  batch_size=config['batch_size'])

        model     = model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=config['epochs'])

        best_val_loss  = float('inf')
        best_state     = None
        patience_count = 0

        for epoch in range(1, config['epochs'] + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()

            if epoch % 10 == 0:
                print(f"  ep {epoch:3d} | tr {tr_loss:.4f}/{tr_acc:.3f} "
                      f"| va {va_loss:.4f}/{va_acc:.3f}")

            # Early stopping
            if va_loss < best_val_loss - 1e-4:
                best_val_loss  = va_loss
                best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= config['patience']:
                    print(f"  Early stop at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        _, best_val_acc = eval_epoch(model, val_loader, criterion, device)
        fold_results.append({'val_acc': best_val_acc, 'val_loss': best_val_loss, 'model': model})
        print(f"  Best val acc: {best_val_acc:.4f}")

    accs = [r['val_acc'] for r in fold_results]
    print(f"\nCV result: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    return fold_results


# =============================================================================
# Model registry
# =============================================================================

MODEL_REGISTRY = {
    'DeepConvNet'  : lambda: DeepConvNet(N_CHANNELS, N_CLASSES),
    'ShallowConvNet': lambda: ShallowConvNet(N_CHANNELS, N_CLASSES),
    'EEGNet'       : lambda: EEGNet(N_CHANNELS, N_CLASSES, N_TIMEPOINTS, SFREQ),
    'ATCNet'       : lambda: ATCNet(N_CHANNELS, N_CLASSES, SFREQ),
    'Conformer'    : lambda: Conformer(N_CHANNELS, N_CLASSES, N_TIMEPOINTS),
    'CTNet'        : lambda: CTNet(N_CHANNELS, N_CLASSES, N_TIMEPOINTS, SFREQ),
    'MBMANet'      : lambda: MBMANet(N_CHANNELS, N_CLASSES, N_TIMEPOINTS, SFREQ),
    'LMDANet'      : lambda: LMDANet(N_CHANNELS, N_CLASSES, N_TIMEPOINTS),
    'EEGITNet'     : lambda: EEGITNet(N_CHANNELS, N_CLASSES, N_TIMEPOINTS),
}

CONFIG = {
    'epochs'     : EPOCHS,
    'batch_size' : BATCH_SIZE,
    'lr'         : LR,
    'cv_folds'   : CV_FOLDS,
    'device'     : DEVICE,
    'patience'   : 15,          # stop if val loss doesn't improve for 15 epochs
}

# =============================================================================
# Run all models
# =============================================================================

all_results = {}

for name, model_fn in MODEL_REGISTRY.items():
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    fold_results = run_cv(model_fn, X, y, CONFIG)
    all_results[name] = fold_results

# =============================================================================
# Summary table
# =============================================================================

print(f"\n{'Model':<16} {'Mean Acc':>10} {'Std':>8}")
print('-' * 36)
for name, results in all_results.items():
    accs = [r['val_acc'] for r in results]
    print(f"{name:<16} {np.mean(accs):>10.4f} {np.std(accs):>8.4f}")