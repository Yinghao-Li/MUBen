# MUBen Documentation

This is the documentation for [MUBen](https://github.com/Yinghao-Li/MUBen/): Mulecular Uncertainty Benchmark.

The code is built to expose implementation details as much as possible and be easily extendable.
Questions and suggestions are welcome if you find any issues while using our code.


## About

![](./img/f1.summarization.png)

MUBen is a benchmark that aims to investigate the performance of uncertainty quantification (UQ) methods built upon backbone molecular representation models.
It implements 6 backbone models (4 pre-trained and 2 non-pre-trained), 8 UQ methods (8 compatible for classification and 6 for regression), and 14 datasets from [MoleculeNet](https://moleculenet.org/) (8 for classification and 6 for regression).
We are actively expanding the benchmark to include more backbones, UQ methods and datasets.
This is an arduous task, and we welcome contribution or collaboration in any form.

### Backbones
| Backbone Models      | Paper | Official Repo | Our Implementation|
| ----------- | ----------- | ----------- | ----------- |
|*Pre-Trained Backbones* |||
| ChemBERTa |[link](https://arxiv.org/abs/2209.01712) | [link](https://github.com/seyonechithrananda/bert-loves-chemistry) | [link](https://github.com/Yinghao-Li/UncertaintyBenchmark/tree/main/muben/chemberta) | 
| GROVER   | [link](https://arxiv.org/abs/2007.02835) | [link](https://github.com/tencent-ailab/grover)| [link](https://github.com/Yinghao-Li/UncertaintyBenchmark/tree/main/muben/grover)|
|Uni-Mol| [link](https://openreview.net/forum?id=6K2RM6wVqKu) | [link](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol) | [link](https://github.com/Yinghao-Li/UncertaintyBenchmark/tree/main/muben/unimol)|
|TorchMD-NET | [Architecture](https://arxiv.org/abs/2202.02541); [Pre-training](https://arxiv.org/abs/2206.00133) | [link](https://github.com/shehzaidi/pre-training-via-denoising) | [link](https://github.com/Yinghao-Li/UncertaintyBenchmark/tree/main/muben/torchmdnet)|
| *Non-Pre-Trained Backbones* |||
|DNN|-|-|[link](https://github.com/Yinghao-Li/UncertaintyBenchmark/tree/main/muben/dnn)|
|GIN| [link](https://arxiv.org/pdf/1810.00826.pdf) | [pyg](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GIN.html) | [link](https://github.com/Yinghao-Li/UncertaintyBenchmark/tree/main/muben/gin)|

### Uncertainty Quantification Methods
| UQ Method | Classification | Regression | Paper |
| ----------- | ----------- | ----------- | ----------- |
| Deterministic | ✅︎ | ✅︎ | - |
| Temperature Scaling | ✅︎ | - | [link](https://arxiv.org/abs/1706.04599) |
| Focal Loss | ✅︎ | - | [link](https://arxiv.org/abs/1708.02002) |
| Deep Ensembles | ✅︎ | ✅︎ | [link](https://arxiv.org/abs/1612.01474) |
| SWAG | ✅︎ | ✅︎ | [link](https://arxiv.org/abs/1808.05326) |
| Bayes by Backprop | ✅︎ | ✅︎ | [link](https://arxiv.org/abs/1505.05424) |
| SGLD | ✅︎ | ✅︎ | [link](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) |
| MC Dropout | ✅︎ | ✅︎ | [link](https://arxiv.org/abs/1506.02142) |

### Data

Please check [MoleculeNet](https://moleculenet.org/datasets-1) for a detailed description.
We use a subset of the MoleculeNet benckmark, including BBBP, Tox21, ToxCast, SIDER, ClinTox, BACE, MUV, HIV, ESOL, FreeSolv, Lipophilicity, QM7, QM8, QM9.

## Data

!!! info
    A set of partitioned datasets are already included in this repo. You can find them under the `./data/` folder: [[scaffold split](https://github.com/Yinghao-Li/UncertaintyBenchmark/tree/main/data/files)]; [[random split](https://github.com/Yinghao-Li/UncertaintyBenchmark/tree/main/data/files-random)].

We utilize the datasets prepared by [Uni-Mol](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol).
You find the data [here](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol#:~:text=pockets.tar.gz-,molecular%20property,-3.506GB) or directly download it through [this link](https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/finetune/molecular_property_prediction.tar.gz).
We place the unzipped files into `./data/UniMol` by default.
For convenience, you are suggested to rename the `qm7dft`, `qm8dft`, and `qm9dft` folders to `qm7`, `qm8`, and `qm9`.

Afterwards, you can transfer the dataset format into ours by running
```bash
PYTHONPATH="." python ./assist/dataset_build_from_unimol.py
``` 
suppose you are in the project root directory.
You can specify the input (Uni-Mol) and output data directories with `--unimol_data_dir` and `--output_dir` arguments.
The script will convert *all* datasets by default (excluding PCBA).
If you want to specify a subset of datasets, you can specify the argument `--dataset_names` with the target dataset names with lowercase letters.

**Notice**: If you would like to run the Uni-Mol model, you are suggested to keep the original `UniMol` data as we will use the pre-defined molecule conformations.
Otherwise, it is safe to remove the original data.


## Ongoing Works

### Active Learning

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