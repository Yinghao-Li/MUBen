import time
import torch
import numpy as np

__all__ = ["Timer"]


class Timer:
    def __init__(self, device='cpu'):
        self._cuda_enabled = torch.device(device).type == 'cuda'
        self.start = None
        self.end = None
        self.started_flag = False
        self.time_elapsed = 0
        self.time_elapsed_cache = list()

    def init(self, device=None):
        self._cuda_enabled = torch.device(device).type == 'cuda' if device is not None else self._cuda_enabled
        self.start = None
        self.end = None
        self.started_flag = False
        self.time_elapsed = 0
        self.time_elapsed_cache = list()

    def clean_cache(self):
        self.time_elapsed_cache = list()

    @property
    def is_empty(self):
        return True if not self.time_elapsed_cache else False

    @property
    def time_elapsed_avg(self):
        return np.mean(self.time_elapsed_cache) if not self.is_empty else None

    def on_measurement_start(self):
        if self._cuda_enabled:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self.start = time.time()
            self.end = None
        self.started_flag = True
        return self

    def on_measurement_end(self):
        """
        Callback function for the time measurement end event.

        Returns
        -------
        bool: whether the elapsed time is recorded
        """
        if not self.started_flag:
            return False

        if self._cuda_enabled:
            self.end.record()
            torch.cuda.synchronize()
            self.time_elapsed = self.start.elapsed_time(self.end)
        else:
            self.time_elapsed = time.time() - self.start
        self.time_elapsed_cache.append(self.time_elapsed)

        self.started_flag = False
        return True
