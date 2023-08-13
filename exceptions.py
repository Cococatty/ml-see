class InferenceTimeValueError(ValueError):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ParamSizeValueError(ValueError):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
