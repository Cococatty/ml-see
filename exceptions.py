"""Build customized errors for this project
"""


class InferenceTimeAssertError(AssertionError):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ParamSizeAssertError(AssertionError):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
