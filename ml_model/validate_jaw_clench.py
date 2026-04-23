"""
Validates jaw_clench_detector.py against annotated .fif files.
Imports detect() and build_baseline() and runs them exactly as a live
session would, so results here reflect real-world performance.

Run from project root:
    python ml_model/validate_jaw_clench.py
"""

import os
import sys
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jaw_clench_detector import build_baseline, detect, make_filter_state, WINDOW_SEC, EXCLUDE

FILES = [
    'data_collection/annotated_eeg/chengyi_4_8_0.fif',
    'data_collection/annotated_eeg/chengyi_4_8_1.fif',
    'data_collection/annotated_eeg/chengyi_4_9_0.fif',
    'data_collection/annotated_eeg/chengyi_4_9_1.fif',
    'data_collection/annotated_eeg/chengyi_4_9_2.fif',
]

# ── Post-detection state machine parameters ───────────────────────────────────
MIN_DUR_SEC   = 0.3   # ignore detections shorter than this
MERGE_GAP_SEC = 0.5   # merge detections closer than this
MATCH_TOL_SEC = 1.5   # detection within this many seconds of annotation = TP


def run_detection(fif_path: str, baseline: dict) -> tuple:
    """
    Slide detect() across an entire .fif file and return
    (detections, rms_trace, times) for evaluation and plotting.
    """
    raw    = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    sfreq  = raw.info['sfreq']
    eeg_ch = [ch for ch in raw.ch_names if ch not in EXCLUDE]
    data   = raw.get_data(picks=eeg_ch)

    win  = int(WINDOW_SEC * sfreq)
    n_win = data.shape[1] // win

    above_threshold = []
    rms_trace       = []
    times           = []

    # filter full signal once for the rms_trace plot
    from scipy.signal import sosfilt
    filtered_full = sosfilt(baseline['sos'], data, axis=1)
    for i in range(n_win):
        f_win = filtered_full[:, i*win:(i+1)*win]
        rms_trace.append(float(np.sqrt((f_win ** 2).mean())))
        times.append(i * win / sfreq)

    # run detect() with maintained filter state — this is what we're validating
    zi = make_filter_state(baseline, data.shape[0])
    for i in range(n_win):
        window = data[:, i*win:(i+1)*win]
        fired, zi = detect(window, baseline, zi)
        above_threshold.append(fired)

    rms_trace = np.array(rms_trace)
    times     = np.array(times)

    # ── State machine: above_threshold → raw events ───────────────────────────
    raw_events = []
    in_event   = False
    for i, fired in enumerate(above_threshold):
        if fired and not in_event:
            onset    = times[i]
            in_event = True
        elif not fired and in_event:
            raw_events.append((onset, times[i]))
            in_event = False
    if in_event:
        raw_events.append((onset, times[-1]))

    # ── Merge nearby events ───────────────────────────────────────────────────
    merged = []
    for onset, offset in raw_events:
        if merged and (onset - merged[-1][1]) < MERGE_GAP_SEC:
            merged[-1] = (merged[-1][0], offset)
        else:
            merged.append((onset, offset))

    # ── Remove short events ───────────────────────────────────────────────────
    detections = [(on, off) for on, off in merged if (off - on) >= MIN_DUR_SEC]

    return detections, rms_trace, times


def evaluate(detections: list, annotations) -> dict:
    gt = [(ann['onset'], ann['onset'] + ann['duration'])
          for ann in annotations if ann['description'] == 'jaw_clench']

    tp_det, tp_gt = set(), set()
    for gi, (g_on, g_off) in enumerate(gt):
        for di, (d_on, d_off) in enumerate(detections):
            if abs(d_on - g_on) < MATCH_TOL_SEC or (d_on < g_off and d_off > g_on):
                tp_det.add(di)
                tp_gt.add(gi)

    tp = len(tp_gt)
    fp = len(detections) - len(tp_det)
    fn = len(gt) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0

    return {'n_gt': len(gt), 'n_det': len(detections),
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1}


def plot_result(raw, detections, rms_trace, times, threshold, title, save_path):
    fig, ax = plt.subplots(figsize=(18, 4))

    ax.plot(times, rms_trace, color='steelblue', linewidth=0.6, alpha=0.8, label='EMG RMS')
    ax.axhline(threshold, color='orange', linewidth=1.5, linestyle='--',
               label=f'Threshold')

    for ann in raw.annotations:
        color = 'green' if ann['description'] == 'jaw_clench' else 'royalblue'
        ax.axvspan(ann['onset'], ann['onset'] + ann['duration'], color=color, alpha=0.2)

    for on, off in detections:
        ax.axvspan(on, off, color='crimson', alpha=0.35)

    patches = [
        mpatches.Patch(color='green',     alpha=0.4, label='GT jaw_clench'),
        mpatches.Patch(color='royalblue', alpha=0.3, label='GT move'),
        mpatches.Patch(color='crimson',   alpha=0.4, label='Detected'),
    ]
    ax.legend(handles=[ax.lines[0], ax.lines[1]] + patches, fontsize=8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RMS (V)')
    ax.set_title(title)
    ax.set_xlim(0, raw.times[-1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


# =============================================================================
# Main
# =============================================================================

os.makedirs('ml_model/jaw_clench_plots', exist_ok=True)
all_metrics = []

for path in FILES:
    fname = os.path.basename(path)
    print(f"\n{'='*55}\n{fname}\n{'='*55}")

    # build baseline from the same file (first 60s)
    baseline = build_baseline(path)

    raw        = mne.io.read_raw_fif(path, preload=True, verbose=False)
    detections, rms_trace, times = run_detection(path, baseline)
    metrics    = evaluate(detections, raw.annotations)
    all_metrics.append({'file': fname, **metrics})

    print(f"  GT={metrics['n_gt']}  Det={metrics['n_det']}  "
          f"TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")
    print(f"  Precision={metrics['precision']:.3f}  "
          f"Recall={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")

    plot_result(
        raw, detections, rms_trace, times, baseline['threshold'],
        title=f"{fname}  |  {metrics['n_det']} detections  "
              f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}",
        save_path=f"ml_model/jaw_clench_plots/{fname.replace('.fif','')}.png"
    )
    print(f"  Saved → ml_model/jaw_clench_plots/{fname.replace('.fif','')}.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}\nSUMMARY\n{'='*55}")
print(f"{'File':<25} {'GT':>4} {'Det':>4} {'TP':>4} {'FP':>4} {'FN':>4} "
      f"{'Prec':>6} {'Rec':>6} {'F1':>6}")
for m in all_metrics:
    print(f"{m['file']:<25} {m['n_gt']:>4} {m['n_det']:>4} {m['tp']:>4} "
          f"{m['fp']:>4} {m['fn']:>4} {m['precision']:>6.3f} "
          f"{m['recall']:>6.3f} {m['f1']:>6.3f}")

print(f"\n  Mean:  "
      f"Precision={np.mean([m['precision'] for m in all_metrics]):.3f}  "
      f"Recall={np.mean([m['recall'] for m in all_metrics]):.3f}  "
      f"F1={np.mean([m['f1'] for m in all_metrics]):.3f}")
