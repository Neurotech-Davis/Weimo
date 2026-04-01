from preprocess_pipeline import (
    MNEPipeline, notch, bandpass, rereference,
    add_idle_class, add_fake_jaw_clench, epoch,
    extract_csp, extract_tensor, extract_cwt, extract_bandpower, extract_pca
)
import numpy as np
import os
import json
import itertools
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Configuration (Step 0: Ensure that the data is downloaded from the google drive and add_annotation.py is ran on that data)
# =============================================================================

files = [
    './data_collection/annotated_eeg/chengyi0210.fif',
    './data_collection/annotated_eeg/pilapil0226.fif',
]
label_map = {'idle': 0, 'move': 1, 'jaw_clench': 2}

# --- Grid axes ---

BANDPASS_CONFIGS = [
    [(8, 13)],               # alpha only
    [(13, 30)],              # beta only
    [(8, 13), (13, 30)],     # alpha + beta
    [(4, 8), (8, 13), (13, 30)],  # theta + alpha + beta
]

FEATURE_EXTRACTORS = {
    'csp':       lambda: extract_csp(n_components=6),
    'bandpower': lambda: extract_bandpower(),
    'pca':       lambda: extract_pca(n_components=10),
    # tensor and cwt produce high-dim arrays; skip for sklearn grid (need flattening)
    # uncomment if you add a flattening wrapper:
    # 'tensor':    lambda: extract_tensor(),
    # 'cwt':       lambda: extract_cwt(),
}

MODEL_GRID = [
    {
        'name': 'LDA',
        'variants': [
            {'solver': 'svd'},
            {'solver': 'lsqr', 'shrinkage': 'auto'},
        ],
        'build': lambda p: LinearDiscriminantAnalysis(**p),
    },
    {
        'name': 'SVC_linear',
        'variants': [{'C': c} for c in [0.1, 1.0, 10.0]],
        'build': lambda p: SVC(kernel='linear', **p),
    },
    {
        'name': 'SVC_rbf',
        'variants': [{'C': c, 'gamma': g} for c, g in itertools.product([0.1, 1.0, 10.0], ['scale', 'auto'])],
        'build': lambda p: SVC(kernel='rbf', **p),
    },
    {
        'name': 'LogisticRegression',
        'variants': [{'C': c, 'max_iter': 1000} for c in [0.1, 1.0, 10.0]],
        'build': lambda p: LogisticRegression(**p),
    },
    {
        'name': 'RandomForest',
        'variants': [{'n_estimators': n, 'max_depth': d} for n, d in itertools.product([100, 300], [None, 5])],
        'build': lambda p: RandomForestClassifier(**p, random_state=42),
    },
    {
        'name': 'HistGBT',
        'variants': [{'learning_rate': lr, 'max_iter': 200} for lr in [0.05, 0.1, 0.2]],
        'build': lambda p: HistGradientBoostingClassifier(**p, random_state=42),
    },
]

CV_FOLDS = 5

# =============================================================================
# Step 1: Load and preprocess raw data for each bandpass config
# =============================================================================

def load_raw_for_bandpass(files, bands, label_map):
    """Run the MNE pipeline for a given bandpass config and return (raw_list)."""
    all_raw = []
    for file in files:
        pipeline = MNEPipeline(file)
        pipeline.add(notch(60))
        pipeline.add(bandpass(bands))
        pipeline.add(rereference())
        pipeline.add(add_idle_class(window_dur=3.0, idle_start_min=1.0))
        pipeline.add(add_fake_jaw_clench())
        pipeline.add(epoch(tmin=0, tmax=3))
        raw_processed = pipeline.run()
        all_raw.append(raw_processed)
    return all_raw


def extract_and_remap(raw_list, extractor_fn, label_map):
    """Apply a feature extractor to each raw object and concatenate."""
    all_X, all_y = [], []
    for raw in raw_list:
        try:
            X, y = extractor_fn()(raw)
        except Exception as e:
            print(f"  [!] Extractor failed: {e}")
            return None, None

        # Flatten non-2D features (e.g. tensor, cwt) for sklearn compatibility
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        auto_ids = raw._event_id
        remap = {v: label_map[k] for k, v in auto_ids.items() if k in label_map}
        y = np.array([remap[yi] for yi in y])
        all_X.append(X)
        all_y.append(y)

    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


# =============================================================================
# Step 2: Grid search
# =============================================================================

results = []

for bands in BANDPASS_CONFIGS:
    band_key = str(bands)
    print(f"\n{'='*60}")
    print(f"Bandpass: {band_key}")
    print(f"{'='*60}")

    raw_list = load_raw_for_bandpass(files, bands, label_map)

    for feat_name, extractor_fn in FEATURE_EXTRACTORS.items():
        print(f"\n  Feature: {feat_name}")

        X, y = extract_and_remap(raw_list, extractor_fn, label_map)
        if X is None:
            continue
        print(f"  X shape: {X.shape}, classes: {np.unique(y)}")

        for model_cfg in MODEL_GRID:
            for params in model_cfg['variants']:
                clf = SklearnPipeline([
                    ('scaler', StandardScaler()),
                    ('clf', model_cfg['build'](params)),
                ])

                cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
                try:
                    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                except Exception as e:
                    print(f"    [!] {model_cfg['name']} {params} failed: {e}")
                    continue

                result = {
                    'bands':     band_key,
                    'features':  feat_name,
                    'model':     model_cfg['name'],
                    'params':    str(params),
                    'mean_acc':  round(float(scores.mean()), 4),
                    'std_acc':   round(float(scores.std()),  4),
                }
                results.append(result)
                print(f"    {model_cfg['name']:20s} {str(params):45s}  "
                      f"acc={scores.mean():.3f} ± {scores.std():.3f}")

# =============================================================================
# Step 3: Save and summarize
# =============================================================================

results.sort(key=lambda r: r['mean_acc'], reverse=True)

print(f"\n{'='*60}")
print("TOP 10 CONFIGURATIONS")
print(f"{'='*60}")
for r in results[:10]:
    print(f"  {r['mean_acc']:.3f} ± {r['std_acc']:.3f}  |  "
          f"bands={r['bands']}  feat={r['features']}  "
          f"model={r['model']}  params={r['params']}")

with open('gridsearch_sklearn_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} results to gridsearch_results.json")