"""
# Author: Yinghao Li
# Modified: September 11th, 2023
# ---------------------------------------
# Description: 

Trainer States Utility

This module provides a data class to handle the inner state of a trainer during 
the training process, including attributes like learning rate, epoch counts, and more. 
The class also provides serialization methods to save and load the trainer state in JSON format.

# Reference: Modified from transformers.trainer_callback: 
https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py

"""


import json
from dataclasses import asdict, dataclass

__all__ = ["TrainerState"]


@dataclass
class TrainerState:
    """
    A data class containing the inner state of a [`Trainer`].

    Parameters
    ----------
    train_log_idx : int
        Counter for training log entries.
    eval_log_idx : int
        Counter for evaluation log entries.
    n_eval_no_improve : int
        Counts how many consecutive evaluation steps
        where model performance hasn't improved.
    valid_epoch_interval : int
        Validation epoch interval.
    lr : float
        Current learning rate.
    lr_scheduler_type : str
        Type of learning rate scheduler.
    n_epochs : int
        Total number of epochs for training.
    model_name : str
        Name of the model, useful for ensemble methods.
    result_dir : str
        Directory to save results, useful for ensemble methods.
    result_dir_no_uncertainty : str
        Directory to save results without uncertainty.
    """

    train_log_idx: int = 1
    eval_log_idx: int = 1
    n_eval_no_improve: int = 0

    valid_epoch_interval: int = 1

    lr: float = None
    lr_scheduler_type: str = "constant"
    n_epochs: int = None
    model_name: str = "model"

    result_dir: str = None
    result_dir_no_uncertainty: str = None

    def init(self):
        """
        Initialize or reset counters related to training and evaluation.
        """
        self.train_log_idx = 1
        self.eval_log_idx = 1
        self.n_eval_no_improve = 0

    def save_to_json(self, json_path: str):
        """
        Save the state of the trainer in JSON format to a file.

        Parameters
        ----------
        json_path : str
            Path to the file where the state should be saved.
        """
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)

    @classmethod
    def load_from_json(cls, json_path: str) -> "TrainerState":
        """
        Load the trainer state from a JSON file.

        Parameters
        ----------
        json_path : str
            Path to the file from which the state should be loaded.

        Returns
        -------
        TrainerState
            An instance of the TrainerState class with attributes set
            based on the loaded JSON content.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))
