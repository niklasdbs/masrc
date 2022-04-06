class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.5):
        self._patience = patience
        self._min_delta = min_delta
        self._validation_counter = 0
        self._best_value = None

    def __call__(self, metric) -> bool:
        if self._best_value is None:
            self._best_value = metric
        elif metric - self._best_value > self._min_delta:
            self._best_value = metric
            self._validation_counter = 0
        elif metric - self._best_value < self._min_delta:
            self._validation_counter += 1
            if self._validation_counter >= self._patience:
                return True
        return False
