"""
gridsearch_torch.py
Grid search over deep learning EEG models.

Axes:
  - Bandpass filter configs
  - Model architecture + hyperparameters

All models consume (B, 1, n_channels, n_timepoints) tensors via extract_tensor().
Evaluation: stratified k-fold cross-validation, reporting mean ± std accuracy.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# =============================================================================
# Patch models.py globals BEFORE importing any model classes.
# The models file uses module-level constants (n_channels, n_classes, etc.)
# that are baked in at class-definition time — we must override them first.
# =============================================================================

N_CHANNELS   = 7    # ← actual number of EEG channels in your data
N_CLASSES    = 3    # idle, move, jaw_clench
SFREQ        = 256  # ← update to your actual sample rate
TRIAL_DUR    = 3.0  # seconds (tmax - tmin in epoch())
N_TIMEPOINTS = int(SFREQ * TRIAL_DUR)

import ml_model.models as _models_module
_models_module.n_channels    = N_CHANNELS
_models_module.n_classes     = N_CLASSES
_models_module.sampling_rate = SFREQ
_models_module.n_timepoints  = N_TIMEPOINTS

# Now safe to import the classes
from ml_model.models import (
    DeepConvNet, ShallowConvNet, EEGNet, ATCNet,
    MBMANet, LMDANet,
    InceptionBlock, TCN,            # building blocks reused below
    ResidualAdd, MultiHeadAttention, FeedForwardBlock,  # reused in CTNet fix
    # CTNet   — broken: uses wrong TransformerEncoder; replaced below
    # EEGITNet — broken: forward() cut off in models.py; replaced below
)

import torch
import torch.nn as nn

# =============================================================================
# CTNet: the models file defines TWO TransformerEncoder classes.
# The one CTNet needs takes (depth, emb_size) but the one resolved at import
# time is the PBT version which takes (n_blocks, d_model, n_head, ...).
# We redefine CTNet here using the correct Conformer-style TransformerEncoder.
# =============================================================================

class _ConformerTransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=10, drop_p=0.5,
                 forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion,
                                 drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            )),
        )

class _ConformerTransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(
            *[_ConformerTransformerEncoderBlock(emb_size) for _ in range(depth)]
        )

class CTNet(nn.Module):
    def __init__(self, num_temporal_filters=8, D=2, dropout=0.5, d=16):
        super().__init__()
        F1  = num_temporal_filters
        F1D = F1 * D
        _t  = N_TIMEPOINTS
        _t  = _t - (SFREQ // 4) + 1   # after temporal conv (no padding)
        _t  = _t // 4                  # after AvgPool(1,4)
        _t  = _t - 16 + 1              # after spatial conv (1,16)
        Tc  = _t // 2                  # after AvgPool(1,2)

        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, SFREQ // 4), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1D, kernel_size=(N_CHANNELS, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1D),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout),
            nn.Conv2d(F1D, d, kernel_size=(1, 16), bias=False),
            nn.BatchNorm2d(d),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout),
        )
        self.transformer = _ConformerTransformerEncoder(depth=6, emb_size=d)
        self.classifier  = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(d * Tc, N_CLASSES),
        )

    def forward(self, x):
        x        = self.conv_layer(x)
        cnn_out  = x.squeeze(2).permute(0, 2, 1)
        trans_out = self.transformer(cnn_out)
        x = (cnn_out + trans_out).flatten(1)
        return self.classifier(x)


# =============================================================================
# EEGITNet: forward() is missing from models.py (document was cut off).
# Reconstructed from the architecture description in the file.
# =============================================================================

class EEGITNet(nn.Module):
    def __init__(self, n_inception=2, out_per_branch=8, D=2,
                 tcn_channels=16, tcn_layers=2, dropout=0.5):
        super().__init__()
        inc_ch = out_per_branch * 3

        blocks = [InceptionBlock(1, out_per_branch)]
        for _ in range(n_inception - 1):
            blocks.append(InceptionBlock(inc_ch, out_per_branch))
        self.inception = nn.Sequential(*blocks)

        self.spatial = nn.Sequential(
            nn.Conv2d(inc_ch, inc_ch * D, kernel_size=(N_CHANNELS, 1),
                      groups=inc_ch, bias=False),
            nn.BatchNorm2d(inc_ch * D),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout),
        )

        sep_ch = inc_ch * D
        self.temporal = nn.Sequential(
            nn.Conv2d(sep_ch, sep_ch, kernel_size=(1, 16), groups=sep_ch,
                      padding='same', bias=False),
            nn.Conv2d(sep_ch, sep_ch, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(sep_ch),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=dropout),
        )

        Tc = N_TIMEPOINTS // 32
        self.tcn        = TCN(sep_ch, tcn_channels, n_layers=tcn_layers, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(tcn_channels * Tc, N_CLASSES),
        )

    def forward(self, x):
        x = self.inception(x)   # (B, inc_ch, N_CHANNELS, T)
        x = self.spatial(x)     # (B, inc_ch*D, 1, T//4)
        x = self.temporal(x)    # (B, sep_ch, 1, T//32)
        x = x.squeeze(2)        # (B, sep_ch, T//32)
        x = self.tcn(x)         # (B, tcn_channels, T//32)
        return self.classifier(x)


# =============================================================================
# Remaining imports
# =============================================================================

from preprocess_pipeline import (
    MNEPipeline, notch, bandpass, rereference,
    add_idle_class, add_fake_jaw_clench, epoch, extract_tensor,
)
import numpy as np
import json
import itertools
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold

# =============================================================================
# Config
# =============================================================================

FILES = [
    './data_collection/annotated_eeg/chengyi0210.fif',
    './data_collection/annotated_eeg/pilapil0226.fif',
]
LABEL_MAP = {'idle': 0, 'move': 1, 'jaw_clench': 2}

BANDPASS_CONFIGS = [
    [(8, 13)],
    [(13, 30)],
    [(8, 13), (13, 30)],
    [(4, 8), (8, 13), (13, 30)],
]

EPOCHS     = 30
BATCH_SIZE = 32
LR         = 1e-3
CV_FOLDS   = 5
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}  |  N_CHANNELS: {N_CHANNELS}  |  N_TIMEPOINTS: {N_TIMEPOINTS}")

# =============================================================================
# Model grid
# =============================================================================

def make_model_grid():
    grid = []

    grid.append(('ShallowConvNet', {},
        lambda: ShallowConvNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)))

    grid.append(('DeepConvNet', {},
        lambda: DeepConvNet()))

    for f1, d in itertools.product([8, 16], [2]):
        _f1, _d = f1, d
        grid.append(('EEGNet', {'F1': _f1, 'D': _d},
            lambda f=_f1, d=_d: EEGNet(num_temporal_filters=f, num_spatial_filters=d)))

    for nf, heads in itertools.product([16, 32], [2]):
        _nf, _h = nf, heads
        grid.append(('ATCNet', {'num_filters': _nf, 'num_heads': _h},
            lambda nf=_nf, h=_h: ATCNet(num_filters=nf, num_heads=h)))

    for f1, dropout in itertools.product([8, 16], [0.3, 0.5]):
        _f1, _dr = f1, dropout
        grid.append(('CTNet', {'F1': _f1, 'dropout': _dr},
            lambda f=_f1, dr=_dr: CTNet(num_temporal_filters=f, dropout=dr)))

    for depth, kernel in itertools.product([9], [25, 51]):
        _depth, _kernel = depth, kernel
        grid.append(('LMDANet', {'depth': _depth, 'kernel': _kernel},
            lambda dep=_depth, k=_kernel: LMDANet(depth=dep, kernel=k)))

    for n_inc, dropout in itertools.product([2, 3], [0.3, 0.5]):
        _ni, _dr = n_inc, dropout
        grid.append(('EEGITNet', {'n_inception': _ni, 'dropout': _dr},
            lambda ni=_ni, dr=_dr: EEGITNet(n_inception=ni, dropout=dr)))

    grid.append(('MBMANet', {}, lambda: MBMANet()))

    return grid


# =============================================================================
# Data loading
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


# =============================================================================
# Training / evaluation
# =============================================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (out.argmax(1) == yb).sum().item()
        n          += len(yb)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb)
        total_loss += criterion(out, yb).item() * len(yb)
        correct    += (out.argmax(1) == yb).sum().item()
        n          += len(yb)
    return total_loss / n, correct / n


def cross_validate(factory_fn, X_tensor, y_tensor, n_splits=CV_FOLDS):
    y_np    = y_tensor.numpy()
    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    dataset = TensorDataset(X_tensor, y_tensor)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_np)), y_np)):
        model     = factory_fn().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=BATCH_SIZE)

        best_val_acc, patience_counter = 0.0, 0
        for ep in range(EPOCHS):
            train_epoch(model, train_loader, optimizer, criterion)
            _, val_acc = eval_epoch(model, val_loader, criterion)
            scheduler.step()
            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break

        fold_accs.append(best_val_acc)
        print(f"    fold {fold+1}/{n_splits}: best_val_acc={best_val_acc:.3f}")

    return fold_accs


# =============================================================================
# Grid search
# =============================================================================

results    = []
model_grid = make_model_grid()

for bands in BANDPASS_CONFIGS:
    band_key = str(bands)
    print(f"\n{'='*60}\nBandpass: {band_key}\n{'='*60}")

    try:
        X_np, y_np = load_tensor_data(FILES, bands, LABEL_MAP)
    except Exception as e:
        print(f"  [!] Data loading failed: {e}")
        continue

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.long)

    for model_name, params, factory_fn in model_grid:
        print(f"\n  Model: {model_name}  params={params}")
        try:
            fold_accs = cross_validate(factory_fn, X_tensor, y_tensor)
        except Exception as e:
            print(f"  [!] Failed: {e}")
            continue

        result = {
            'bands':     band_key,
            'model':     model_name,
            'params':    str(params),
            'mean_acc':  round(float(np.mean(fold_accs)), 4),
            'std_acc':   round(float(np.std(fold_accs)),  4),
            'fold_accs': [round(a, 4) for a in fold_accs],
        }
        results.append(result)
        print(f"  → {model_name} {params}  "
              f"acc={result['mean_acc']:.3f} ± {result['std_acc']:.3f}")

# =============================================================================
# Save + summarize
# =============================================================================

results.sort(key=lambda r: r['mean_acc'], reverse=True)

print(f"\n{'='*60}\nTOP 10 CONFIGURATIONS\n{'='*60}")
for r in results[:10]:
    print(f"  {r['mean_acc']:.3f} ± {r['std_acc']:.3f}  |  "
          f"bands={r['bands']}  model={r['model']}  params={r['params']}")

with open('gridsearch_torch_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} results to gridsearch_torch_results.json")