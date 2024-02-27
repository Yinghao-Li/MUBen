"""
# Author: Yinghao Li
# Modified: August 26th, 2023
# ---------------------------------------
# Description:

Simple training timer to measure and record the execution time 
of different sections of the training process. It supports both 
CUDA and CPU timings.
"""

import time
import torch
import numpy as np

__all__ = ["Timer"]


class Timer:
    def __init__(self, device="cpu"):
        """
        Initialize the Timer object.

        Parameters
        ----------
        device : str, optional
            Device type to determine if CUDA is enabled, default is "cpu".
        """
        self._cuda_enabled = torch.device(device).type == "cuda"
        self.start = None
        self.end = None
        self.started_flag = False
        self.time_elapsed = 0
        self.time_elapsed_cache = list()

    def init(self, device=None):
        """
        Initialize or reinitialize the timer's properties.

        Parameters
        ----------
        device : str, optional
            Device type to determine if CUDA is enabled. If not provided,
            retains the existing setting.
        """
        self._cuda_enabled = (
            torch.device(device).type == "cuda"
            if device is not None
            else self._cuda_enabled
        )
        self.start = None
        self.end = None
        self.started_flag = False
        self.time_elapsed = 0
        self.time_elapsed_cache = list()

    def clean_cache(self):
        """
        Reset the elapsed time cache.
        """
        self.time_elapsed_cache = list()

    @property
    def is_empty(self):
        """Check if the cache is empty.

        Returns
        -------
        bool
            True if cache is empty, otherwise False.
        """
        return True if not self.time_elapsed_cache else False

    @property
    def time_elapsed_avg(self):
        """
        Calculate the average of elapsed times in the cache.

        Returns
        -------
        float or None
            The average elapsed time if cache is not empty, otherwise None.
        """
        return np.mean(self.time_elapsed_cache) if not self.is_empty else None

    def on_measurement_start(self):
        """
        Start the time measurement. Utilizes CUDA events if CUDA is enabled,
        otherwise uses Python's time module.

        Returns
        -------
        Timer
            The current Timer object.
        """
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
        End the time measurement and record the elapsed time.

        Returns
        -------
        bool
            True if the elapsed time is successfully recorded, otherwise False.
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
