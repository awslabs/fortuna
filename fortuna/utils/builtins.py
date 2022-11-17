class HashableMixin:
    def __hash__(self) -> int:
        return hash(
            tuple(
                [
                    getattr(self, k)
                    for k in sorted(vars(self).keys())
                    if not k.startswith("_")
                ]
            )
        )

    def __eq__(self, other) -> bool:
        self_keys = [k for k in vars(self).keys() if not k.startswith("_")]
        other_keys = [k for k in vars(other).keys() if not k.startswith("_")]

        same_keys = self_keys == other_keys
        if same_keys and isinstance(other, self.__class__):
            same_vals = all(
                map(lambda k: getattr(self, k) == getattr(other, k), self_keys)
            )
            return same_vals
        return False
