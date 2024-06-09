import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy import signal as ss
import numpy as np
import os
import re

SAMPLE_FREQUENCY = 2000  # (Hz)
default_notch_freq = 60.0  # Frequency to be removed from signal (Hz) - Determine this with Fourier logic above (peaks)
default_q_factor = 30.0


def _fft(sig):
    yf = rfft(sig)
    xf = rfftfreq(len(sig), 1 / SAMPLE_FREQUENCY)
    return pd.Series(data=np.abs(yf), index=xf)


def plot_fourier_xform(signals_df: pd.DataFrame, channels: list[str] = None) -> None:
    for ch in channels or signals_df.columns:
        vals = _fft(signals_df[ch].values)
        vals.plot(title=ch, xlim=(0, 200), figsize=(20, 4), color="r")
        plt.show()


# https://dsp.stackexchange.com/questions/49460/apply-low-pass-butterworth-filter-in-python
def butter_filter(signal: np.ndarray, pps: int, freq_cutoff: int = 30):
    w = freq_cutoff / (pps / 2)  # Normalize the frequency
    b, a = ss.butter(5, w, "low", analog=False, output="ba")
    return ss.filtfilt(b, a, signal)


########### Configs ###############


def denoise(signal, notch_freq: float, Q: float):
    _sig = signal.values if type(signal) == pd.Series else signal
    b, a = ss.iirnotch(notch_freq, Q, SAMPLE_FREQUENCY)
    return ss.filtfilt(b, a, _sig)


def _filter_frequencies(signal: np.array, notch_frequencies: list[float], sampling_freq: int):
    for notch_freq in notch_frequencies:
        b, a = ss.iirnotch(notch_freq, int(notch_freq / 2), sampling_freq)
        signal = ss.filtfilt(b, a, signal)
    return signal.astype("int32")


def plot_denoised(
    sig, notch_freq: float = default_notch_freq, Q: float = default_q_factor, title_prefix: str = "", y_axis=(-0.5, 0.5)
):
    t = range(0, len(sig))
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(40, 6), dpi=300)
    plt.subplots_adjust(left=0.025, right=0.9, top=0.9, bottom=0.1)
    ax[0].set_title(title_prefix + " Original")
    ax[0].set(ylim=y_axis)
    ax[0].plot(sig, color="r")

    ax[1].set_title(f"{title_prefix} With Q: {Q}")
    ax[1].set(ylim=y_axis)
    ax[1].plot(t, denoise(sig, notch_freq, Q))

    return denoise(sig, notch_freq, Q)


def write_denoised_bins(
    export_dir: str, channels: list[str], notch_freq: float = default_notch_freq, Q: float = default_q_factor
) -> list[str]:
    """Write new versions of BIN export files with noisy bandwidths removed.  New BIN files are written to the 'local' folder within the project root directory.

    Args:
        export_dir (str): folder containing the raw BIN files
        channels (list[str]): list of noisy channels that need to be cleaned
        notch_freq (float, optional): notch frequency to be removed. Defaults to default_notch_freq.
        Q (float, optional): Quality factor for the frequency removal. Defaults to default_q_factor.

    Returns:
        list[str]: list of BIN files that we copied and modified
    """
    all_bins = []
    for channel in channels:
        bin_re = re.compile(f"{channel}.+[.]BIN")

        bin_files = [f for f in os.listdir(export_dir) if bin_re.match(f)]

        for bfile in bin_files:
            out_file = f"{bfile[:-4]}.BIN"
            sig = np.fromfile(os.path.join(export_dir, bfile), dtype=np.int32)
            denoised = denoise(sig, notch_freq, Q)
            denoised.astype("int32").tofile("local/" + out_file)

        all_bins += bin_files

    return all_bins
