# MUBen: **M**olecular **U**ncertainty **Ben**mark
Code associated with paper *MUBen: Benchmarking the Uncertainty of Pre-Trained Models for Molecular Property Prediction* [[arXiv](https://arxiv.org/abs/2306.10060)].

Please visit [Documentation](https://yinghao-li.github.io/MUBen/) for the full documentation of this project, which contains more comprehensive API introductions and examples.

![](./figures/f1.summarization.png)


MUBen is a benchmark that aims to investigate the performance of uncertainty quantification (UQ) methods built upon backbone molecular representation models.
It implements 6 backbone models (4 pre-trained and 2 non-pre-trained), 8 UQ methods (8 compatible for classification and 6 for regression), and 14 datasets from [MoleculeNet](https://moleculenet.org/) (8 for classification and 6 for regression).
We are actively expanding the benchmark to include more backbones, UQ methods, and datasets.
This is an arduous task, and we welcome contribution or collaboration in any form.

## Backbones
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

## Uncertainty Quantification Methods
| UQ Method | Classification | Regression | Paper |
| ----------- | ----------- | ----------- | ----------- |
| *Included* |||
| Deterministic | ✅︎ | ✅︎ | - |
| Temperature Scaling | ✅︎ | - | [link](https://arxiv.org/abs/1706.04599) |
| Focal Loss | ✅︎ | - | [link](https://arxiv.org/abs/1708.02002) |
| Deep Ensembles | ✅︎ | ✅︎ | [link](https://arxiv.org/abs/1612.01474) |
| SWAG | ✅︎ | ✅︎ | [link](https://arxiv.org/abs/1808.05326) |
| Bayes by Backprop | ✅︎ | ✅︎ | [link](https://arxiv.org/abs/1505.05424) |
| SGLD | ✅︎ | ✅︎ | [link](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) |
| MC Dropout | ✅︎ | ✅︎ | [link](https://arxiv.org/abs/1506.02142) |

## Data

Please check [MoleculeNet](https://moleculenet.org/datasets-1) for a detailed description.
We use a subset of the MoleculeNet benchmark, including BBBP, Tox21, ToxCast, SIDER, ClinTox, BACE, MUV, HIV, ESOL, FreeSolv, Lipophilicity, QM7, QM8, QM9.

## CITATION

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
