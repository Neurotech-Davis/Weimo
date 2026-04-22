"""
Compare signal properties between annotated training .fif files and a live-recorded .fif.
Primary concern: magnitude differences that would cause distribution shift at inference.

Usage (from project root):
    python processing/compare_recordings.py <live.fif> [--training-dir data_collection/annotated_eeg]
"""

import sys
import os
import argparse
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

EXCLUDE = {'Trigger', 'Event', 'TRG'}


def load_raw(path):
    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    picks = [ch for ch in raw.ch_names if ch not in EXCLUDE]
    data = raw.get_data(picks=picks)  # (n_channels, n_times)
    return raw, data, picks


def signal_stats(data, label):
    """Print and return basic amplitude statistics."""
    abs_data = np.abs(data)
    stats = {
        'label'   : label,
        'mean'    : data.mean(),
        'std'     : data.std(),
        'rms'     : np.sqrt((data ** 2).mean()),
        'abs_mean': abs_data.mean(),
        'abs_p95' : np.percentile(abs_data, 95),
        'abs_p99' : np.percentile(abs_data, 99),
        'max'     : abs_data.max(),
        'unit'    : 'V (raw)',
    }
    print(f"\n── {label} ──────────────────────────────")
    print(f"  mean     : {stats['mean']:.3e}")
    print(f"  std      : {stats['std']:.3e}")
    print(f"  RMS      : {stats['rms']:.3e}")
    print(f"  |x| mean : {stats['abs_mean']:.3e}")
    print(f"  |x| p95  : {stats['abs_p95']:.3e}")
    print(f"  |x| p99  : {stats['abs_p99']:.3e}")
    print(f"  |x| max  : {stats['max']:.3e}")
    return stats


def per_channel_stats(data, ch_names, label):
    print(f"\n── Per-channel RMS ({label}) ──────────────")
    for i, ch in enumerate(ch_names):
        rms = np.sqrt((data[i] ** 2).mean())
        print(f"  {ch:<6} RMS={rms:.3e}")


def psd_comparison(datasets, sfreq, save_path):
    """Overlay PSD of training files vs live recording."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {'training': 'steelblue', 'live': 'crimson'}

    for label, data, color in datasets:
        # average PSD across channels
        psds = []
        for ch in range(data.shape[0]):
            psd, freqs = mne.time_frequency.psd_array_welch(
                data[ch:ch+1], sfreq=sfreq, fmin=1, fmax=100, n_fft=min(data.shape[1], 1024), verbose=False
            )
            psds.append(psd[0])
        mean_psd = np.mean(psds, axis=0)
        ax.semilogy(freqs, mean_psd, color=color, label=label, alpha=0.8, linewidth=1.5)

    ax.axvspan(8, 13, alpha=0.08, color='green', label='alpha (8-13 Hz)')
    ax.axvspan(13, 30, alpha=0.08, color='orange', label='beta (13-30 Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (V²/Hz)')
    ax.set_title('PSD comparison: training vs live recording (mean across channels)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"\nSaved PSD plot → {save_path}")
    plt.show()


def amplitude_distribution(datasets, save_path):
    """Histogram of raw signal amplitudes — most direct view of magnitude shift."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    colors = {'training': 'steelblue', 'live': 'crimson'}

    for label, data, color in datasets:
        flat = data.flatten()
        # raw amplitude
        axes[0].hist(flat, bins=200, density=True, alpha=0.5, color=color, label=label)
        # absolute value
        axes[1].hist(np.abs(flat), bins=200, density=True, alpha=0.5, color=color, label=label)

    axes[0].set_title('Raw amplitude distribution')
    axes[0].set_xlabel('Amplitude (V)')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    axes[1].set_title('|Amplitude| distribution')
    axes[1].set_xlabel('|Amplitude| (V)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Saved amplitude plot → {save_path}")
    plt.show()


def ratio_summary(training_stats_list, live_stats):
    """Print the ratio of live to training for key stats — the key diagnostic."""
    keys = ['rms', 'abs_mean', 'abs_p95', 'abs_p99']
    train_means = {k: np.mean([s[k] for s in training_stats_list]) for k in keys}

    print("\n── Magnitude ratio: live / training mean ──────────────")
    print("  (ratio >> 1 means live signal is much larger → likely distribution shift)")
    print("  (ratio << 1 means live signal is much smaller)")
    for k in keys:
        ratio = live_stats[k] / train_means[k]
        flag  = '  ⚠ LARGE DIFFERENCE' if ratio > 3 or ratio < 0.33 else ''
        print(f"  {k:<12}: {ratio:.2f}x{flag}")


def main():
    parser = argparse.ArgumentParser(description='Compare training vs live EEG recordings')
    parser.add_argument('live_fif', help='Path to live-recorded .fif file')
    parser.add_argument('--training-dir', default='data_collection/annotated_eeg',
                        help='Directory containing annotated training .fif files')
    parser.add_argument('--out-dir', default='processing/comparison_plots',
                        help='Directory to save output plots')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load live recording ───────────────────────────────────────────────────
    print(f"\nLoading live recording: {args.live_fif}")
    live_raw, live_data, live_ch = load_raw(args.live_fif)
    live_sfreq = live_raw.info['sfreq']
    live_stats  = signal_stats(live_data, f'LIVE  ({os.path.basename(args.live_fif)})')
    per_channel_stats(live_data, live_ch, 'live')

    # ── Load training files ───────────────────────────────────────────────────
    training_files = sorted([
        os.path.join(args.training_dir, f)
        for f in os.listdir(args.training_dir)
        if f.endswith('.fif')
    ])
    if not training_files:
        print(f"No .fif files found in {args.training_dir}")
        sys.exit(1)

    print(f"\nLoading {len(training_files)} training files from {args.training_dir}")
    training_datasets = []
    training_stats_list = []

    for path in training_files:
        _, data, ch = load_raw(path)
        label = os.path.basename(path)
        stats = signal_stats(data, f'TRAIN ({label})')
        training_stats_list.append(stats)
        training_datasets.append((label, data, 'steelblue'))

    # ── Ratio summary ─────────────────────────────────────────────────────────
    ratio_summary(training_stats_list, live_stats)

    # ── Check channel name match ──────────────────────────────────────────────
    _, _, train_ch = load_raw(training_files[0])
    print(f"\n── Channel check ─────────────────────────────────────")
    print(f"  Training channels : {train_ch}")
    print(f"  Live channels     : {live_ch}")
    if set(train_ch) != set(live_ch):
        print("  ⚠ Channel mismatch — model may receive wrong spatial arrangement")
    else:
        print("  ✓ Channels match")

    # ── Check sample rate ─────────────────────────────────────────────────────
    train_sfreq = mne.io.read_raw_fif(training_files[0], verbose=False).info['sfreq']
    print(f"\n── Sample rate check ─────────────────────────────────")
    print(f"  Training sfreq : {train_sfreq} Hz")
    print(f"  Live sfreq     : {live_sfreq} Hz")
    if train_sfreq != live_sfreq:
        print("  ⚠ Sample rate mismatch")
    else:
        print("  ✓ Sample rates match")

    # ── Plots ─────────────────────────────────────────────────────────────────
    datasets_for_plot = training_datasets + [('live', live_data, 'crimson')]

    # collapse training into one array for cleaner plots
    all_train_data = np.concatenate([d for _, d, _ in training_datasets], axis=1)
    plot_datasets = [('training (all sessions)', all_train_data, 'steelblue'),
                     ('live', live_data, 'crimson')]

    psd_comparison(plot_datasets, live_sfreq,
                   os.path.join(args.out_dir, 'psd_comparison.png'))
    amplitude_distribution(plot_datasets,
                           os.path.join(args.out_dir, 'amplitude_distribution.png'))


if __name__ == '__main__':
    main()
