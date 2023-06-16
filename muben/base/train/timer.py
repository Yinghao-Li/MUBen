import time
import torch

__all__ = ["Timer"]


class Timer:
    def __init__(self, device='cpu'):
        self._cuda_enabled = torch.device(device).type == 'cuda'
        self.start = None
        self.end = None
        self.started_flag = False

    def init(self, device=None):
        self._cuda_enabled = torch.device(device).type == 'cuda' if device is not None else self._cuda_enabled
        self.start = None
        self.end = None
        self.started_flag = False

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
        if not self.started_flag:
            return None

        if self._cuda_enabled:
            self.end.record()
            torch.cuda.synchronize()
            time_elapsed = self.start.elapsed_time(self.end)
        else:
            time_elapsed = time.time() - self.start

        self.started_flag = False
        return time_elapsed
