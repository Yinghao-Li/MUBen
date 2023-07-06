"""
# Author: Yinghao Li
# Created: July 6th, 2023
# Modified: July 6th, 2023
# ---------------------------------------
# Description: Build the ROE dataset (not included).
"""

import os.path as op
import pandas as pd
from muben.utils.io import save_json


def save_csv(instances, output_path):
    df_ = pd.DataFrame(instances)
    df_.to_csv(output_path)


file_path = '../data/roe.csv'
result_dir = '../data/files/roe/'
df = pd.read_csv(file_path)[['smiles_monomer', '1/length', 'source', 'ROE (kj/mol)']]
df = df[df['1/length'] == 0]

dft_df = df[df['source'] == 'DFT']
exp_df = df[df['source'] == 'expt']

training_data = {
    'smiles': dft_df['smiles_monomer'],
    'labels': [[lb] for lb in dft_df['ROE (kj/mol)']],
    'masks': [[1]] * len(dft_df)
}
valid_df = exp_df.sample(frac=0.4)
test_df = exp_df.drop(valid_df.index)
valid_data = {
    'smiles': valid_df['smiles_monomer'],
    'labels': [[lb] for lb in valid_df['ROE (kj/mol)']],
    'masks': [[1]] * len(valid_df)
}
test_data = {
    'smiles': test_df['smiles_monomer'],
    'labels': [[lb] for lb in test_df['ROE (kj/mol)']],
    'masks': [[1]] * len(test_df)
}

save_csv(training_data, op.join(result_dir, 'train.csv'))
save_csv(valid_data, op.join(result_dir, 'valid.csv'))
save_csv(test_data, op.join(result_dir, 'test.csv'))


meta_dict = {
    'task_type': 'regression',
    'n_tasks': 1,
    'classes': None,
    'eval_metric': 'rmse'
}
save_json(meta_dict, op.join(result_dir, "meta.json"), collapse_level=2)
