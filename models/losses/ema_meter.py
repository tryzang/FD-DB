class EMAMeter:
    def __init__(self, decay: float = 0.98, eps: float = 1e-6):
        self.decay = float(decay)
        self.eps = float(eps)
        self.value = None
        self.min = None
        self.max = None

    def update(self, value: float) -> float:
        value = float(value)
        if self.value is None:
            self.value = value
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * value
        if self.min is None:
            self.min = self.value
        else:
            self.min = min(self.min, self.value)
        if self.max is None:
            self.max = self.value
        else:
            self.max = max(self.max, self.value)
        return self.value

    def ratio(self) -> float:
        if self.value is None or self.min is None:
            return 1.0
        return float(self.value) / (float(self.min) + self.eps)

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "ratio": self.ratio(),
            "min": self.min,
            "max": self.max,
        }
