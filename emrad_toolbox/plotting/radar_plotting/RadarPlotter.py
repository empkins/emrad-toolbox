from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pywt
from PyEMD import EMD
from scipy import fft
from scipy.signal import spectrogram, stft

from emrad_toolbox.plotting.radar_plotting.PlottingExceptions import WaveletCoefficientsNotProvidedError


class RadarPlotter:
    """A class used to plot various types of radar signals."""

    @staticmethod
    def plot_radar_magnitude(radar_signal, sampling_rate, signal_type: str = "", ax=None):
        """
        Plot the magnitude of a radar signal over time.

        Parameters
        ----------
        :param radar_signal : The radar signal to plot.
        :param sampling_rate : The sampling rate of the radar signal.
        :param signal_type : The type of the signal. Defaults to an empty string.
        :param ax : The axes object to draw the plot on. If None, a new figure and axes are
        created.
        """
        if np.any(np.iscomplex(radar_signal)):
            radar_signal = np.abs(radar_signal)
        t = np.arange(0, len(radar_signal) / sampling_rate, 1 / sampling_rate)

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(t, radar_signal)
        ax.set_title(f"Magnitude {signal_type}")
        ax.set_ylabel("")
        ax.set_xlabel("Time [sec]")

    @staticmethod
    def plot_wavelet(  # noqa: PLR0913
        radar_signal: np.array,
        sampling_rate: float,
        plot_magnitude: bool = False,
        wavelet_type: str = "morl",
        wavelet_coefficients: Tuple[int, int] = (1, 256),
        signal_type: str = "",
        ax=None,
    ):
        """
        Plot the wavelet transform of a radar signal.

        Parameters
        ----------
        :param radar_signal: The radar signal to plot.
        :param sampling_rate : The sampling rate of the radar signal.
        :param plot_magnitude: If True, plots the magnitude of the radar signal. Defaults to False.
        :param wavelet_type : The type of the wavelet to use for the transform. Defaults to 'morl'.
        :param wavelet_coefficients : The range of scales to use for the wavelet transform.
        Defaults to (1, 256).
        :param signal_type : The type of the signal. Defaults to an empty string.
        :param ax: The axes object to draw the plot on. If None, a new figure and axes are created.
        """
        if wavelet_coefficients is None:
            raise WaveletCoefficientsNotProvidedError()

        if plot_magnitude:
            radar_signal = np.abs(radar_signal)

        scales = np.arange(wavelet_coefficients[0], wavelet_coefficients[1])
        coefficients, frequencies = pywt.cwt(radar_signal, scales, wavelet_type)
        time = np.arange(0, len(radar_signal) / sampling_rate, 1 / sampling_rate)

        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots()
        cax = ax.imshow(
            np.abs(coefficients),
            aspect="auto",
            cmap="jet",
            extent=[time.min(), time.max(), frequencies.min(), frequencies.max()],
        )
        fig.colorbar(cax, ax=ax, label="Magnitude")
        ax.set_yscale("log")
        ax.set_ylabel("Scale (Inverse of Frequency)")
        ax.set_xlabel("Time (seconds)")
        ax.set_title(f"Wavelet Transform of Magnitude with {wavelet_type} wavelet {signal_type}")

    @staticmethod
    def plot_fft_spectrogram(radar_signal, sampling_rate, decibel_as_unit: bool = True, signal_type: str = "", ax=None):
        """
        Plot the FFT spectrogram of a radar signal.

        Parameters
        ----------
        :param radar_signal : The radar signal to plot.
        :param sampling_rate : The sampling rate of the radar signal.
        :param decibel_as_unit: If True, uses decibels as the unit for the spectrogram. Defaults to True.
        :param signal_type : The type of the signal. Defaults to an empty string.
        :param ax : The axes object to draw the plot on. If None, a new figure and axes
        are created.
        """
        if np.any(np.iscomplex(radar_signal)):
            radar_signal = np.abs(radar_signal)
        f, t, sxx = spectrogram(radar_signal, sampling_rate, return_onesided=False)
        label = "Magnitude "
        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots()

        sxx_shifted = fft.fftshift(sxx)
        if decibel_as_unit:
            sxx_shifted = 10 * np.log10(sxx_shifted)
            label += "[dB]"
        cax = ax.pcolormesh(t, fft.fftshift(f), sxx_shifted, shading="nearest")
        ax.set_title(f"Spectrogram (FFT {signal_type})")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [sec]")
        ax.set_ylim([-80, 80])
        fig.colorbar(cax, ax=ax, label=label)

    @staticmethod
    def plot_fft_phase_spectrogram(
        radar_signal, sampling_rate, decibel_as_unit: bool = True, signal_type: str = "", ax=None
    ):
        """
        Plot the FFT phase spectrogram of a radar signal.

        Parameters
        ----------
        :param radar_signal : The radar signal to plot.
        :param sampling_rate : The sampling rate of the radar signal.
        :param decibel_as_unit : If True, uses decibels as the unit for the spectrogram. Defaults to True.
        :param signal_type : The type of the signal. Defaults to an empty string.
        :param ax : The axes object to draw the plot on. If None, a new figure and axes are created.
        """
        RadarPlotter.plot_fft_spectrogram(np.angle(radar_signal), sampling_rate, decibel_as_unit, signal_type, ax)

    @staticmethod
    def plot_stft_spectrogram(  # noqa: PLR0913
        radar_signal,
        sampling_rate,
        nperseg: int = 512,
        noverlap: int = 384,
        y_lim: Optional[List[int]] = None,
        c_lim: Optional[List[int]] = None,
        dB_as_unit: bool = True,  # noqa: N803
        signal_type: str = "",
        ax=None,
    ):
        """
        Plot the STFT spectrogram of a radar signal.

        Parameters
        ----------
        :param radar_signal : The radar signal to plot.
        :param sampling_rate : The sampling rate of the radar signal.
        :param nperseg : The length of each segment for the STFT. Defaults to 512.
        :param noverlap : The number of points to overlap between segments. Defaults to 384.
        :param y_lim : The limits for the y-axis. Defaults to None.
        :param c_lim : The limits for the colorbar. Defaults to None.
        :param dB_as_unit : If True, uses decibels as the unit for the spectrogram. Defaults to True.
        :param signal_type : The type of the signal. Defaults to an empty string.
        :param ax: The axes object to draw the plot on. If None, a new figure and axes are created.
        """
        label = "Magnitude "
        if dB_as_unit:
            label += "[dB]"
        frequencies, times, zxx = stft(
            radar_signal,
            sampling_rate,
            return_onesided=np.any(np.iscomplex(radar_signal)),
            nperseg=nperseg,
            noverlap=noverlap,
        )
        zxx_shifted = np.fft.fftshift(zxx, axes=0)
        frequencies_shifted = np.fft.fftshift(frequencies)
        if dB_as_unit:
            zxx_shifted = 10 * np.log10(np.abs(zxx_shifted))
        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots()
        cax = ax.pcolormesh(times, frequencies_shifted, zxx_shifted, shading="nearest")
        ax.set_title(f"STFT Spectrogram {signal_type}")
        if y_lim is not None and len(y_lim) == 2:
            ax.set_ylim(y_lim)
        if c_lim is not None and len(c_lim) == 2:
            cax.set_clim(c_lim)
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [sec]")
        fig.colorbar(cax, ax=ax, label=label)
        plt.show()

    @staticmethod
    def plot_derivative_stft_spectrogram(  # noqa: PLR0913
        radar_signal,
        sampling_rate,
        nperseg: int = 512,
        noverlap: int = 384,
        y_lim: Optional[List[int]] = None,
        c_lim: Optional[List[int]] = None,
        decibel_as_unit: bool = True,
        signal_type: str = "",
        ax=None,
    ):
        """
        Plot the derivative STFT spectrogram of a radar signal.

        Parameters
        ----------
        :param radar_signal: The radar signal to plot.
        :param sampling_rate : The sampling rate of the radar signal.
        :param nperseg : The length of each segment for the STFT. Defaults to 512.
        :param noverlap : The number of points to overlap between segments. Defaults to 384.
        :param y_lim : The limits for the y-axis. Defaults to None.
        :param c_lim : The limits for the colorbar. Defaults to None.
        :param decibel_as_unit : If True, uses decibels as the unit for the spectrogram. Defaults to True.
        :param signal_type : The type of the signal. Defaults to an empty string.
        :param ax: The axes object to draw the plot on. If None, a new figure and axes are created.
        """
        RadarPlotter.plot_stft_spectrogram(
            np.gradient(radar_signal), sampling_rate, nperseg, noverlap, y_lim, c_lim, decibel_as_unit, signal_type, ax
        )

    @staticmethod
    def plot_emd(
        radar_signal,
        sampling_rate,
        max_imfs: int = -1,
        signal_type: str = "",
    ):
        """
        Plot the Empirical Mode Decomposition (EMD) of a radar signal.

        Parameters
        ----------
        :param radar_signal : The radar signal to plot.
        :param sampling_rate : The sampling rate of the radar signal.
        :param max_imfs : The maximum number of Intrinsic Mode Functions (IMFs) to compute.
        If -1, all IMFs are computed. Defaults to -1.
        :param signal_type : The type of the signal. Defaults to an empty string.
        """
        if np.any(np.iscomplex(radar_signal)):
            radar_signal = np.abs(radar_signal)
        emd = EMD()
        imfs = emd.emd(radar_signal, np.arange(len(radar_signal)) / sampling_rate, max_imf=max_imfs)
        num_imfs = imfs.shape[0]
        num_samples = imfs.shape[1]
        time = np.linspace(0, num_samples / sampling_rate, num_samples)
        fig, axs = plt.subplots(num_imfs, 1, figsize=(10, 2 * num_imfs))
        fig.suptitle(f"EMD {signal_type}")

        for i in range(num_imfs):
            axs[i].plot(time, imfs[i])
            axs[i].set_title(f"IMF {i + 1}")
        fig.tight_layout()
