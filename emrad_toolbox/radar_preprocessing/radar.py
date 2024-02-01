"""emrad_toolbox radar - A collection of tools to handle Radar data, such as applying filters."""

from typing import Dict, Tuple

import numpy as np
from numba import njit
from scipy.signal import butter, decimate, filtfilt, hilbert, sosfilt


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
        filter_cutoff: Tuple[int, int] = (80, 18),
        filter_order: int = 4,
        fs: int = 1000,
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
        fs: int = 1000,
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
        fs: int = 1000,
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
    def calculate_displacement_vector(i: np.array, q: np.array, fs: float = 61e9, c_mps: float = 299708516) -> np.array:
        """Calculate the displacement vector of the complex signal.

        :param i: The in-phase component of the complex signal.
        :param q: The quadrature component of the complex signal.
        :param fs: The sampling frequency of the signal.
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

        radian_2_meter = c_mps / (4 * np.pi * fs)
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
