# Environment Setup

!!! warning
    Our code is developed with `Python 3.10`.
    It does **not** work with `Python < 3.9`.


## Installation

MUBen is available as a Python package on [PyPI](https://pypi.org/project/muben/) and can be installed using pip.
If you prefer to use MUBen as a standalone package and do not need to modify the source code, you can simply run:
```bash
pip install muben
```

!!! warning
    The `muben` package from PyPI does not include datasets or post-processing functions.
    To access these, you need to download the files manually or clone the repository.

For those who wish to modify the source code or conduct experiments using the complete capabilities of MUBen, we recommend cloning the repository.
You can fork the project on GitHub and clone your fork, or directly clone the original repository:

```bash
# Clone your fork of the repository
git clone https://github.com/<your GitHub username>/MUBen.git

# Or clone the original repository with git
git clone https://github.com/Yinghao-Li/MUBen.git --single-branch --branch main
```


## Anaconda

Using Anaconda is the most straightforward approach to start a new virtual environment for a project.
Suppose you have anaconda or miniconda installed in your local machine, you can create a new `conda` environment for this project using `conda create`.
```bash
conda create -n muben python=3.10
```

The required packages are listed in `requirements.txt`.
It is recommended to install these dependencies with `pip install` as `conda install` may sometimes encounter dependency resolution issue.
```bash
conda activate muben
pip install -r requirements.txt
```

## Docker
You can also run this project within a docker container.
The docker image can be built through `docker build`.
```bash
docker build -t muben ./docker
```
And `docker run` is the command to start your container in the terminal.
```bash
docker run --gpus all -it --rm  muben
```
Please check the [official documentation](https://docs.docker.com/) for more options.

## Backbone Checkpoints

Some backbone models require loading pre-trained model checkpoints.

- For ChemBERTa, we use the `DeepChem/ChemBERTa-77M-MLM` checkpoint hosted on Hugging Face's [Model Hub](https://huggingface.co/models). You can specify the model name to the argument `--pretrained_model_name_or_path` (which is set to default), or you can download the model and pass the path to the model to the argument.
- The `GROVER-base` checkpoint is available at GROVER's [project repo](https://github.com/tencent-ailab/grover) or can be directly downloaded through [this link](https://ai.tencent.com/ailab/ml/ml-data/grover-models/pretrain/grover_base.tar.gz).
Unzip the downloaded `.tar.gz` file to get the `.pt` checkpoint.
- The `Uni-Mol` checkpoint is available at Uni-Mol's [project repo](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol) or can be directly downloaded through [this link](https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt).
- The `TorchMD-NET` checkpoint is available at this [project repo](https://github.com/shehzaidi/pre-training-via-denoising) or can be directly downloaded through [this link](https://github.com/shehzaidi/pre-training-via-denoising/raw/main/checkpoints/denoised-pcqm4mv2.ckpt).

<!-- By default, the code will look for the models at locations `./models/grover_base.pt` and `./models/unimol_base.pt`, respectively. -->
For GROVER, Uni-Mol and TorchMD-NET, You need to specify the `--checkpoint_path` argument to the path to your downloaded checkpoints.

