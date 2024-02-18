"""emrad_toolbox radar - A collection of tools to handle Radar data, such as applying filters."""

from typing import Dict, List, Tuple

import numpy as np
import pywt
from numba import njit
from scipy.signal import butter, decimate, filtfilt, find_peaks, hilbert, sosfilt
from vmdpy import VMD

from emrad_toolbox.radar_preprocessing.preprocessing_exceptions import (
    ScalingFactorsNotProvidedError,
)


class RadarPreprocessor:
    """RadarPreprocessor - A collection of static function to handle Radar data, such as applying filters."""

    @staticmethod
    @njit
    def calculate_angle(i: np.array, q: np.array) -> np.array:
        """Calculate the angle of the complex signal.

        :parameter i: The in-phase component of the complex signal.
        :parameter q: The quadrature component of the complex signal.
        :return: The angle of the complex signal.
        """
        tan = np.unwrap(np.arctan2(i, q))
        angle = np.diff(tan, axis=0, prepend=tan[0])
        return angle

    @staticmethod
    @njit
    def calculate_power(i: np.array, q: np.array) -> np.array:
        """Calculate the power of the complex signal.

        :param i: The in-phase component of the complex signal.
        :param q: The quadrature component of the complex signal.
        :return: The power of the complex signal as a numpy array.
        """
        return np.sqrt(np.square(i.astype("float64")) + np.square(q.astype("float64")))

    @staticmethod
    def butterworth_band_pass_filter(
        i: np.array,
        q: np.array,
        filter_cutoff: Tuple[int, int] = (18, 80),
        filter_order: int = 4,
        fs: float = 1000,
    ) -> Dict[str, np.array]:
        """Apply a Butterworth band pass filter to the complex signal with the specified order.

        :param i: The in-phase component of the complex signal.
        :param q: The quadrature component of the complex signal.
        :param filter_cutoff: A tuple containing the lower and upper cutoff frequency of the band pass filter.
        :param filter_order: The order of the filter.
        :param fs: The sampling frequency of the signal.
        :return: A dictionary containing the filtered in-phase and quadrature components as well as the magnitude.
        """
        filtering_dict = {}
        b, a = butter(
            N=filter_order,
            Wn=[filter_cutoff[0], filter_cutoff[1]],
            btype="band",
            analog=False,
            fs=fs,
        )
        filtering_dict["I_band_pass"] = filtfilt(b, a, i, axis=0)
        filtering_dict["Q_band_pass"] = filtfilt(b, a, q, axis=0)
        filtering_dict["Magnitude_band_pass"] = RadarPreprocessor.calculate_power(
            filtering_dict["I_band_pass"], filtering_dict["Q_band_pass"]
        )
        return filtering_dict

    @staticmethod
    def butterworth_high_pass_filtering(
        i: np.array,
        q: np.array,
        high_pass_filter_cutoff_hz: float = 0.1,
        high_pass_filter_order: int = 4,
        fs: float = 1000,
    ) -> Dict[str, np.array]:
        """Apply a Butterworth high pass filter to the complex signal with the specified order.

        :param i: The in-phase component of the complex signal.
        :param q: The quadrature component of the complex signal.
        :param high_pass_filter_cutoff_hz: The cutoff frequency of the high pass filter.
        :param high_pass_filter_order: The order of the filter.
        :param fs: The sampling frequency of the signal.
        :return:
        """
        filtering_dict = {}
        b, a = butter(
            N=high_pass_filter_order,
            Wn=high_pass_filter_cutoff_hz,
            btype="high",
            analog=False,
            fs=fs,
        )
        filtering_dict["I_high_pass"] = filtfilt(b, a, i, axis=0).astype("int64")
        filtering_dict["Q_high_pass"] = filtfilt(b, a, q, axis=0).astype("int64")
        filtering_dict["Magnitude_high_pass"] = np.sqrt(
            np.square(filtering_dict["I_high_pass"]) + np.square(filtering_dict["Q_high_pass"])
        )
        return filtering_dict

    @staticmethod
    def envelope(average_length: int = 100, magnitude: np.array = None) -> np.array:
        """Calculate the envelope of the complex signal.

        :param average_length: The length of the averaging window.
        :param magnitude: The magnitude of the complex signal.
        :return: The envelope of the complex signal.
        """
        if magnitude is None:
            raise ValueError()
        return np.convolve(
            np.abs(hilbert(magnitude)).flatten(),
            np.ones(average_length) / average_length,
            mode="same",
        )

    @staticmethod
    def butterworth_low_pass_filtering(
        i: np.array,
        q: np.array,
        low_pass_filter_cutoff_hz: float = 10,
        low_pass_filter_order: int = 4,
        fs: float = 1000,
    ):
        """Apply a Butterworth low pass filter to the complex signal with the specified order.

        :param i: The in-phase component of the complex signal.
        :param q: The quadrature component of the complex signal.
        :param low_pass_filter_cutoff_hz: The cutoff frequency of the low pass filter.
        :param low_pass_filter_order: The order of the filter.
        :param fs: Tje sampling frequency of the signal.
        :return: A dictionary containing the filtered in-phase and quadrature components.
        """
        b, a = butter(
            N=low_pass_filter_order,
            Wn=low_pass_filter_cutoff_hz,
            btype="low",
            analog=False,
            fs=fs,
        )
        filtering_dict = {
            "I_low_pass": filtfilt(b, a, i, axis=0),
            "Q_low_pass": filtfilt(b, a, q, axis=0),
        }
        filtering_dict["Magnitude_low_pass"] = np.sqrt(
            np.square(filtering_dict["I_low_pass"]) + np.square(filtering_dict["Q_low_pass"])
        )
        return filtering_dict

    @staticmethod
    def downsample(downsampling_factor: int = 20, data_to_downsample: np.array = None) -> np.array:
        """Downsample the data by the specified factor.

        :param downsampling_factor: Factor by which the data should be downsampled.
        :param data_to_downsample: Data to be downsampled.
        :return: The downsampled data.
        """
        if data_to_downsample is None:
            raise ValueError()
        return decimate(data_to_downsample, downsampling_factor, axis=0)

    @staticmethod
    def calculate_displacement_vector(
        i: np.array, q: np.array, radar_frequency: float = 61e9, c_mps: float = 299708516
    ) -> np.array:
        """Calculate the displacement vector of the complex signal.

        :param i: The in-phase component of the complex signal.
        :param q: The quadrature component of the complex signal.
        :param radar_frequency: The frequency of the radar signal.
        :param c_mps: The speed of light in meters per second.
        :return: A dictionary containing the displacement vector.
        """
        # coordinates of the barycenter
        i_m = np.mean(i.astype("float64"))
        q_m = np.mean(q.astype("float64"))

        # reduce to local WCS
        u = i - i_m
        v = q - q_m

        #    s_uu * uc +  s_uv * vc = (s_uuu + s_uvv)/2
        #    s_uv * uc +  s_vv * vc = (s_uuv + s_vvv)/2
        s_uv = np.sum(u * v)
        s_uu = np.sum(u**2)
        s_vv = np.sum(v**2)
        s_uuv = np.sum(u**2 * v)
        s_uvv = np.sum(u * v**2)
        s_uuu = np.sum(u**3)
        s_vvv = np.sum(v**3)

        # Solving the linear system
        a = np.array([[s_uu, s_uv], [s_uv, s_vv]])
        b = np.array([s_uuu + s_uvv, s_vvv + s_uuv]) / 2.0
        uc, vc = np.linalg.solve(a, b)

        ic_1 = i_m + uc  # coordinates of the barycenter
        qc_1 = q_m + vc

        # Radius and Residual
        ri_1 = np.sqrt((i - ic_1) ** 2 + (q - qc_1) ** 2)
        r_1 = np.mean(ri_1)
        residu_1 = np.sum((ri_1 - r_1) ** 2)

        ic = i - ic_1
        qc = q - qc_1

        angle = np.arctan2(ic, qc)

        radian_2_meter = c_mps / (4 * np.pi * radar_frequency)
        pw_unwrapped = np.unwrap(angle) * radian_2_meter
        return {
            "PW_in_m": pw_unwrapped,
            "Residuals_mean_square": residu_1,
            "Receive_signal_strength": r_1,
        }

    @staticmethod
    def calculate_pulse_wave_component(
        displacement_vector: np.array, fs: float, order: int = 4, cut_off_values: Tuple[float, float] = (0.5, 20)
    ) -> np.array:
        """Calculate the pulse wave component from the displacement vector.

        :param displacement_vector: The displacement vector of the complex signal.
        :param fs: The sampling frequency of the signal.
        :param order: The speed of light in meters per second.
        :param cut_off_values: The cutoff values of the filter as a tuple (the lower value has to come first).
        :return: The pulse wave component.
        """
        if cut_off_values[1] < cut_off_values[0]:
            return RadarPreprocessor.calculate_pulse_wave_component(
                displacement_vector, fs, order, (cut_off_values[1], cut_off_values[0])
            )
        sos = butter(N=order, Wn=[cut_off_values[0], cut_off_values[1]], output="sos", fs=fs, btype="bandpass")
        pulse_wave = sosfilt(sos, displacement_vector)
        return pulse_wave

    @staticmethod
    def apply_matched_filter(radar_signal, expected_signal):
        """
        Apply a matched filter to a complex radar signal.

        The matched filter is created by taking the complex conjugate of the expected signal and reversing it in time.
        The radar signal is then convolved with this matched filter.

        :param radar_signal: The complex radar signal to which the matched filter is to be applied.
        :param expected_signal: The expected signal used to create the matched filter.
        :return: The filtered signal after applying the matched filter.
        """
        matched_filter = np.conj(expected_signal[::-1])
        filtered_signal = np.convolve(radar_signal, matched_filter, mode="same")
        return filtered_signal

    @staticmethod
    def _create_trajectory_matrix(radar_signal, window_size):
        rows = []
        for i in range(len(radar_signal) - window_size + 1):
            window = radar_signal[i : i + window_size]
            rows.append(window)
        trajectory_matrix = np.array(rows)
        return trajectory_matrix

    @staticmethod
    def _calculate_svd(trajectory_matrix):
        u, s, vh = np.linalg.svd(trajectory_matrix, full_matrices=False)
        return u, s, vh

    @staticmethod
    def calculate_svd(radar_signal, window_size):
        """
        Calculate the Singular Value Decomposition (SVD) of a radar signal.

        This function creates a trajectory matrix from the radar signal with a specified window size.
        It then calculates the SVD of this trajectory matrix.

        :param radar_signal: The radar signal from which the SVD is to be calculated.
        :param window_size: The size of the window for the trajectory matrix.
        :return: The U, S, and V^H matrices of the SVD.
        """
        trajectory_matrix = RadarPreprocessor._create_trajectory_matrix(radar_signal, window_size)
        u, s, vh = RadarPreprocessor._calculate_svd(trajectory_matrix)
        return u, s, vh

    @staticmethod
    def matched_filter(radar_signal, expected_signal):
        """
        Apply a matched filter to a complex radar signal.

        The matched filter is created by taking the complex conjugate of the expected signal and reversing it in time.
        The radar signal is then convolved with this matched filter.

        :param radar_signal: The complex radar signal to which the matched filter is to be applied.
        :param expected_signal: The expected signal used to create the matched filter.
        :return: The filtered signal after applying the matched filter.
        """
        matched_filter = np.conj(expected_signal[::-1])
        filtered_signal = np.convolve(radar_signal, matched_filter, mode="same")
        return filtered_signal

    @staticmethod
    def svd_and_matched_filter(radar_signal, window_size):
        """
        Calculate the Singular Value Decomposition (SVD) of a radar signal and apply a matched filter.

        This function creates a trajectory matrix from the radar signal with a specified window size.
        It then calculates the SVD of this trajectory matrix.
        The matched filter is then applied to the radar signal.

        :param radar_signal: The complex radar signal to which the SVD and matched filter are to be applied.
        :param window_size: The size of the window for the trajectory matrix.
        :return: The filtered signal after applying the matched filter.
        """
        u, s, vh = RadarPreprocessor.calculate_svd(radar_signal, window_size)
        filtered_signals = []
        for i in range(len(s)):
            filtered_signals.append(RadarPreprocessor.matched_filter(radar_signal, vh[i]))
        return filtered_signals

    @staticmethod
    def calculate_optimum_scaling_factor(
        radar_signal: np.array, sampling_rate: float = 1953.125, scaling_factor_range: List[int] = [0, 100]
    ) -> Tuple[int, np.array]:
        """
        Calculate the optimum scaling factor for the Continuous Wavelet Transform (CWT) of a radar signal.

        The algorithm behind this function is by Tomii and Ohtsuki (2015)

        :param radar_signal:
        :param sampling_rate:
        :param scaling_factor_range:
        :return: scaling factor
        """
        # Check if radar_signal is complex, if so calculate the power of the signal
        if np.any(np.iscomplex(radar_signal)):
            radar_signal = np.abs(radar_signal)
        if scaling_factor_range is None or len(scaling_factor_range) != 2:
            raise ScalingFactorsNotProvidedError()

        peaks = np.zeros(scaling_factor_range[1] - scaling_factor_range[0] + 1, dtype=int)
        sampling_period = len(radar_signal) / sampling_rate
        min_distance = 0.5 * sampling_rate
        if sampling_period < 0.3:
            raise ValueError("TimePeriodTooSmallError")
        gathered_coefficients = []
        for i in range(scaling_factor_range[0], scaling_factor_range[1] + 1):
            # Peak detection with min interval 500ms
            coefficients, frequencies = pywt.cwt(
                radar_signal, scales=i, wavelet="morl", sampling_period=sampling_period, method="conv"
            )
            gathered_coefficients.append(coefficients)
            a = len(find_peaks(coefficients[0], distance=min_distance)[0])
            peaks[i - scaling_factor_range[0]] = int(a)
        # find the most frequent value in the peaks
        p_m = np.bincount(peaks).argmax()
        i = scaling_factor_range[0]
        while peaks[i - scaling_factor_range[0]] == p_m and i < scaling_factor_range[1]:
            i += 1
        return (
            i,
            gathered_coefficients[i - scaling_factor_range[0]][0],
        )

    @staticmethod
    def apply_vmd(radar_signal, alpha, tau, k, dc, init, tol):  # noqa: PLR0913
        """
        Apply Variational Mode Decomposition on a complex signal.

        :param radar_signal (np.array): The complex signal to decompose.
        :param alpha (float): The balancing parameter of the data-fidelity constraint.
        :param tau (float): The noise-tolerance (no strict fidelity enforcement).
        :param k (int): The number of modes to be recovered.
        :param dc (bool): Whether to include DC mode.
        :param init (int): The initialization method.
        :param tol (float): The tolerance of convergence criterion.

        :return u (np.array): The decomposed modes.
        :return u_hat (np.array): The spectrum of the modes.
        :return omega (np.array): The center frequencies of the modes.
        """
        u, u_hat, omega = VMD(radar_signal, alpha, tau, k, dc, init, tol)
        return u, u_hat, omega
