"""
# Author: Yinghao Li
# Modified: August 23rd, 2023
# ---------------------------------------
# Description: Trainer states, modified from transformers.trainer_callback:
               https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py
"""


import json
from dataclasses import asdict, dataclass

__all__ = ["TrainerState"]


@dataclass
class TrainerState:
    """
    A class containing the [`Trainer`] inner state.
    """

    train_log_idx: int = 0  # will increase by 1 each time you call `train_epoch`
    eval_log_idx: int = 0  # will increase by 1 each time you call `eval_and_save`
    # counts how many evaluation steps in total the model performance has not improved
    n_eval_no_improve: int = 0

    valid_epoch_interval: int = 1

    lr: float = None
    lr_scheduler_type: str = "constant"
    n_epochs: int = None
    model_name: str = "model"  # mutable model name for ensemble

    result_dir: str = None  # mutable result directory for ensemble
    result_dir_no_uncertainty: str = None  # substitute uncertainty method to none

    def init(self):
        self.train_log_idx: int = 0
        self.eval_log_idx: int = 0
        self.n_eval_no_improve: int = 0

    def save_to_json(self, json_path: str):
        """
        Save the content of this instance in JSON format inside `json_path`.
        """
        json_string = (
            json.dumps(
                {k: v for k, v in asdict(self).items() if "timer" not in k},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """
        Create an instance from the content of `json_path`.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))
