class ScalingFactorsNotProvidedError(Exception):
    """Exception raised when the Range for the scaling factors is not provided."""

    def __init__(self, message="Scaling Factor range must be provided, e.g. [0,100]"):
        self.message = message
        super().__init__(self.message)


class TimePeriodTooSmallError(Exception):
    """Exception raised when the sampling period is below 500 Milliseconds."""

    def __init__(self, message="Sampling Period must be at least 500 Milliseconds"):
        self.message = message
        super().__init__(self.message)
