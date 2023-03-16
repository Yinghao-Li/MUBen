import os
import numpy as np
import pandas as pd


def save_results(lbs, preds: np.ndarray, file_path: str):
    """
    Save results to disk as csv files
    """

    data_dict = dict()

    data_dict['true'] = lbs

    assert len(preds.shape) < 3, ValueError("Cannot save results with larger ")
    if len(preds.shape) == 2:
        preds = [pred_tuple.tolist() for pred_tuple in preds]
    data_dict['pred'] = preds

    os.makedirs(os.path.dirname(os.path.normpath(file_path)), exist_ok=True)

    df = pd.DataFrame(data_dict)
    df.to_csv(file_path)
