class WaveletCoefficientsNotProvidedError(Exception):
    """Exception raised when the wavelet coefficients are not provided."""

    def __init__(self, message="Wavelet coefficients must be provided"):
        self.message = message
        super().__init__(self.message)
