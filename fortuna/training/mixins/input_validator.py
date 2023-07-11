class InputValidatorMixin:
    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            raise AttributeError("Cannot recognize inputs arguments: {}".format(args))
        if len(kwargs) > 0:
            raise AttributeError(
                "{} are not valid input arguments.".format(list(kwargs.keys()))
            )
        super(InputValidatorMixin, self).__init__(*args, **kwargs)
