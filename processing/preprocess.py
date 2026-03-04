import mne
import matplotlib.pyplot as plt
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA


class MNEData():
    def __init__(self, fif_file: str, preload: bool = True):
        self.data = mne.io.read_raw_fif(fif_file, preload=preload)
        self.info = self.data.info.to_json_dict()
        self.eeg_channels = [name for name in self.data.ch_names if "EEG" in name]
        self.preload=preload

    def plot(self, before: mne.io.Raw, after: mne.io.Raw, before_title: str = "Before", after_title: str = "After"):
        """Plot PSD comparison between two Raw objects side by side."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        for ax, data, title in zip(axes, [before, after], [before_title, after_title]):
            psd, freqs = mne.time_frequency.psd_array_welch(
                data.get_data(), fmin=0.1, fmax=100, n_fft=2048, sfreq=self.info['sfreq']
            )
            for i, ch in enumerate(self.eeg_channels):
                ax.semilogy(freqs, psd[i], label=ch, alpha=0.7)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power (V²/Hz)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def filter_power_noise(self, freq: float = 60.0, plot: bool = False):
        """Apply a notch filter to remove power line noise (default 60 Hz)."""
        before = self.data.copy()
        self.data.notch_filter(freq)

        if plot:
            self.plot(
                before, self.data,
                before_title="Before Notch Filter",
                after_title=f"After Notch Filter ({freq} Hz)"
            )

        return self

    def filter_bandpass(self, low_cut: float, high_cut: float, plot: bool = False):
        """Apply a bandpass filter between low_cut and high_cut Hz."""
        before = self.data.copy()
        self.data.filter(l_freq=low_cut, h_freq=high_cut)

        if plot:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            for ax, data, title in zip(axes, [before, self.data], ["Before Bandpass Filter", f"After Bandpass Filter ({low_cut}–{high_cut} Hz)"]):
                psd, freqs = mne.time_frequency.psd_array_welch(
                    data.get_data(), fmin=0.1, fmax=100, n_fft=2048, sfreq=self.info['sfreq']
                )
                for i, ch in enumerate(self.eeg_channels):
                    ax.semilogy(freqs, psd[i], label=ch, alpha=0.7)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Power (V²/Hz)')
                ax.legend()
                ax.grid(True, alpha=0.3)

            axes[1].axvspan(low_cut, high_cut, alpha=0.1, color='green', label='Pass band')
            plt.tight_layout()
            plt.show()

        return self

    def create_bins(self, tmin: float, tmax: float): # TODO
        """Makes a set of window lists for each annotation"""
        events_array, event_id_dict = mne.events_from_annotations(self.data)
        
        self.epochs = mne.Epochs(
            self.data,
            events_array,
            event_id=event_id_dict,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=self.preload
        )

        return self

    def plot_trial(self, bin): # TODO
        """Visualize each trial and a bin. Serves as a visual check for removing noisy trials."""
        pass

    def remove_trial(self, bin): # TODO
        """Removes a list of indices of trials from in a bin"""
        pass

    def extract_features(self): # TODO
        """Returns a np array of vectors for each window representing meaningful features"""
        # freq bands
        # - foureier 

        

        pass

    def pca(self, pca): # TODO
        """Reduces dimensions on the feature vectors"""
        pca_data = pca.fit_transform(self.epochs.get_data())
        self.pca_data = pca_data

        return self

def main():
    '''
    preprocess all the files in a directory or a single file and output the 
    data necessary for the ML model
    input: 
    output: 
    '''
    pca = UnsupervisedSpatialFilter(PCA(n_components=10), average=False)
    data = MNEData(fif_file='./data_collection/annotated_eeg/chengyi0210.fif')
    data.create_bins(tmin=0.0, tmax=0.5)    
    data.pca(pca=pca)

    # (n_trials, n_channels, n_times)
    print(data.data.shape)


if __name__ == "__main__":
    main()
