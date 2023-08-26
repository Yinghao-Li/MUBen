"""
# Author: Yinghao Li
# Modified: August 23rd, 2023
# ---------------------------------------
# Description: Manage model checkpoints.
"""

import copy
import regex
import torch
import logging
import numpy as np
from typing import Optional, Union

from muben.utils.macro import StrEnum

logger = logging.getLogger(__name__)

__all__ = ["UpdateCriteria", "CheckpointContainer"]


class UpdateCriteria(StrEnum):
    metric_smaller = "metric-smaller"
    metric_larger = "metric-larger"
    always = "always"


class CheckpointContainer:
    def __init__(
        self, update_criteria: Optional[Union[str, UpdateCriteria]] = "always"
    ):
        """
        Parameters
        ----------
        update_criteria: decides whether the metrics are in descend order
        """
        assert update_criteria in UpdateCriteria.options(), ValueError(
            f"Invalid criteria! Options are {UpdateCriteria.options()}"
        )
        self._criteria = update_criteria

        self._state_dict = None
        self._metric = (
            np.inf if self._criteria == UpdateCriteria.metric_smaller else -np.inf
        )

    @property
    def state_dict(self):
        return self._state_dict

    @property
    def metric(self):
        return self._metric

    def check_and_update(
        self, model, metric: Optional[Union[int, float]] = None
    ) -> bool:
        """
        Check whether the new model performs better than the buffered models.
        If so, replace the worst model in the buffer by the new model

        Parameters
        ----------
        metric: metric to compare the model performance
        model: the models

        Returns
        -------
        bool, whether there's any change to the buffer
        """
        update_flag = (
            (self._criteria == UpdateCriteria.always)
            or (
                self._criteria == UpdateCriteria.metric_smaller
                and metric <= self.metric
            )
            or (
                self._criteria == UpdateCriteria.metric_larger and metric >= self.metric
            )
        )

        if update_flag:
            self._metric = metric
            model.to("cpu")
            model_cp = copy.deepcopy(model)

            self._state_dict = model_cp.state_dict()
            return True

        return False

    def save(self, model_dir: str):
        """
        Parameters
        ----------
        model_dir: which directory to save the model

        Returns
        -------
        None
        """
        out_dict = dict()
        for attr, value in self.__dict__.items():
            if regex.match(f"^_[a-z]", attr):
                out_dict[attr] = value

        torch.save(out_dict, model_dir)
        return None

    def load(self, model_dir: str):
        """
        Parameters
        ----------
        model_dir: from which directory to load the model

        Returns
        -------
        self
        """
        model_dict = torch.load(model_dir)

        for attr, value in model_dict.items():
            if attr not in self.__dict__:
                logger.warning(
                    f"Attribute {attr} is not natively defined in model buffer!"
                )
            setattr(self, attr, value)

        return self
