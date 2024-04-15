# MUBen Documentation


[![GitHub](https://img.shields.io/badge/GitHub-MUBen-white)](https://github.com/Yinghao-Li/MUBen)
[![OpenReview](https://img.shields.io/badge/%F0%9F%94%97%20OpenReview-TMLR-darkred)](https://openreview.net/forum?id=qYceFeHgm4)
[![arXiv](https://img.shields.io/badge/arXiv-2306.10060-b31b1b.svg)](https://arxiv.org/abs/2306.10060)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Yinghao-Li/MUBen/)
[![PyPI version](https://badge.fury.io/py/muben.svg)](https://badge.fury.io/py/muben)


!!! info
    This is the documentation for **[MUBen](https://github.com/Yinghao-Li/MUBen/): Mulecular Uncertainty Benchmark**.
    The code is built to expose implementation details as much as possible and be easily extendable.
    Questions and suggestions are welcome if you find any issues while using our code.



![](./img/f1.summarization.png)

MUBen is a benchmark that aims to investigate the performance of uncertainty quantification (UQ) methods built upon backbone molecular representation models.
It implements 6 backbone models (4 pre-trained and 2 non-pre-trained), 8 UQ methods (8 compatible for classification and 6 for regression), and 14 datasets from [MoleculeNet](https://moleculenet.org/) (8 for classification and 6 for regression).
We are actively expanding the benchmark to include more backbones, UQ methods and datasets.
This is an arduous task, and we welcome contribution or collaboration in any form.

!!! info
    The following of this page introduces the basic information and structure of the MUBen project.
    For its utilization or customization, please visit the [Experiments](./train.cli.md) or [Customization](./customize.md) pages.

## Backbones

The following backbone models are implemented in MUBen, and their performance discussed in the published article.

| Backbone Models      | Paper | Official Repo |
| ----------- | ----------- | ----------- |
|***Pre-Trained*** |||
| ChemBERTa |[link](https://arxiv.org/abs/2209.01712) | [link](https://github.com/seyonechithrananda/bert-loves-chemistry) | 
| GROVER   | [link](https://arxiv.org/abs/2007.02835) | [link](https://github.com/tencent-ailab/grover)|
|Uni-Mol| [link](https://openreview.net/forum?id=6K2RM6wVqKu) | [link](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol) |
|TorchMD-NET | [Architecture](https://arxiv.org/abs/2202.02541); [Pre-training](https://arxiv.org/abs/2206.00133) | [link](https://github.com/shehzaidi/pre-training-via-denoising) |
| ***Trained from Scratch*** |||
|DNN|-|-|
|GIN| [link](https://arxiv.org/pdf/1810.00826.pdf) | [pyg](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GIN.html) |

MUBen also support easy integration of your own backbone models.
To use your own backbones, please check the [customization guide](./customize.md).

## Uncertainty Quantification Methods

Currently, MUBen supports the following uncertainty quantification methods.
Notice that some methods only work with one of the classification/regression task.

| UQ Method | Classification | Regression | Paper |
| ----------- | ----------- | ----------- | ----------- |
| ***Included in Paper*** |||
| Deterministic | ✅︎ | ✅︎ | - |
| Temperature Scaling | ✅︎ | - | [link](https://arxiv.org/abs/1706.04599) |
| Focal Loss | ✅︎ | - | [link](https://arxiv.org/abs/1708.02002) |
| Deep Ensembles | ✅︎ | ✅︎ | [link](https://arxiv.org/abs/1612.01474) |
| SWAG | ✅︎ | ✅︎ | [link](https://arxiv.org/abs/1808.05326) |
| Bayes by Backprop | ✅︎ | ✅︎ | [link](https://arxiv.org/abs/1505.05424) |
| SGLD | ✅︎ | ✅︎ | [link](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) |
| MC Dropout | ✅︎ | ✅︎ | [link](https://arxiv.org/abs/1506.02142) |
| ***Additional in Repo*** |||
| Evidential Networks |✅︎|✅︎|[link](https://openreview.net/forum?id=xqS8k9E75c)|
| Conformal Prediction |-|✅︎| [link](https://arxiv.org/abs/2107.07511) |
| Isotonic Calibration| - | ✅︎ | [link](https://arxiv.org/abs/1905.06023)|


## Data

!!! info
    The prepared scaffold-split data is available in the [./data/files/](https://github.com/Yinghao-Li/UncertaintyBenchmark/tree/main/data/files) directory on GitHub.

This documentation utilizes a selection from the MoleculeNet benchmark, which includes datasets such as BBBP, Tox21, ToxCast, SIDER, ClinTox, BACE, MUV, HIV, ESOL, FreeSolv, Lipophilicity, QM7, QM8, and QM9.
For detailed descriptions of these datasets, please refer to the [MoleculeNet website](https://moleculenet.org/datasets-1).

We employ the "molecular property" datasets curated by [Uni-Mol](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol), which are accessible for download [here](https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/finetune/molecular_property_prediction.tar.gz).
While the original Uni-Mol dataset is generally not necessary, it is used to provide pre-defined molecule conformations for running the Uni-Mol model.
To use the Uni-Mol data, download and unzip the files into the `./data/UniMol/` directory.
For ease of reference, you are suggested to rename the `qm7dft`, `qm8dft`, and `qm9dft` directories to `qm7`, `qm8`, and `qm9`, respectively.
The conversion of the dataset format from Uni-Mol to our specifications can be viewed in the script [dataset_build_from_unimol.py](https://github.com/Yinghao-Li/MUBen/blob/main/assist/dataset_build_from_unimol.py).

Typically, each dataset comprises 4 files: `train.csv`, `valid.csv`, `test.csv`, and `meta.json`.
The `.csv` files partition the data into training, validation, and testing sets, while `meta.json` contains metadata such as task type (classification or regression), number of tasks, and number of classes (for classification tasks).
Each `.csv` file contains three columns:
- `smiles`: A string representing the SMILES notation of a molecule.
- `labels`: A list of integers or floats representing the property values to be predicted for each molecule. The length of the list corresponds to the number of tasks.
- `masks`: A binary list (containing 0s and 1s) where 1 indicates a valid property value and 0 indicates an invalid value to be ignored during training and testing.

The dataset is automatically loaded during training through the method `muben.dataset.Dataset.prepare()`.
For a practical example, visit the [example](./train.python.md) page.

## Experimental Results

We have made our experimental results available in the [./reports/](https://github.com/Yinghao-Li/MUBen/tree/main/reports) directory on GitHub.
These results are organized into different folders based on the nature of the experiments:

- `primary`: Contains the most comprehensive set of results derived from experiments on scaffold-split datasets.
- `random`: Includes results from experiments conducted on datasets that were split randomly.
- `frozen`: Features results from experiments where the pre-trained model's weights were frozen, except for the last output layer, which was updatable.
- `distribution`: Offers results from the QM9 dataset, where the test set was categorized into five bins based on the average Tanimoto similarities to the training scaffolds.

Files within these directories are named following the pattern `<backbone>-<dataset>.csv`.
Each file provides a comparison of different UQ methods.
The rows detail the performance of each UQ method, while the columns display the mean and standard deviation from three random runs for each metric.

Additional post-processing scripts can be found in the [./assist/](https://github.com/Yinghao-Li/MUBen/tree/main/assist) directory, which include files starting with `plot_` or `results_`.
These scripts are useful for further analysis and visualization of the experimental data.

## Ongoing Works: Active Learning

We are developing code to integrate *active learning* into the pipeline.
Specifically, we assume we have a small set of labeled data points (`--n_init_instances`) at the beginning.
Within each active learning iteration, we use the labeled dataset to fine-tune the model parameters and select a batch of data points (`--n_al_select`) from the unlabeled set with the least predicted certainty (*i.e.*, max predicted entropy for classification and max predicted variance for regression).
The process is repeated for several loops (`--n_al_loops`), and the intermediate performance is tracked.

The code is still under construction and currently is **only available under the `dev` branch**.
In addition, several points are worth attention:

- Currently, only DNN and ChemBERTa backbones are supported (`./run/dnn_al.py` and `./run/chemberta_al.py`). Migrating AL to other backbones is not difficult but requires updating some Trainer functions if they are reloaded.
- To enable active learning, make sure you set `--enable_active_learning` to `true`.
- Currently, Deep Ensembles is not supported for AL.
- We cannot guarantee the correctness of our implementation. If you notice any abnormalities in the code, please do not hesitate to post an issue.

One example is
```bash
python ./run/dnn_al.py \
  --enable_active_learning \
  --n_init_instances 100 \
  --n_al_loops 20 \
  --n_al_select 20 \
  # other model and training hyper-parameters...
```

## Citation

If you find our work helpful, please consider citing it as
```latex
@misc{li2023muben,
    title={MUBen: Benchmarking the Uncertainty of Pre-Trained Models for Molecular Property Prediction},
    author={Yinghao Li and Lingkai Kong and Yuanqi Du and Yue Yu and Yuchen Zhuang and Wenhao Mu and Chao Zhang},
    year={2023},
    eprint={2306.10060},
    archivePrefix={arXiv},
    primaryClass={physics.chem-ph}
}
```