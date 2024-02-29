# Get Started

In the following, we give a brief introduction to our experiment pipeline, which might be helpful if you would like to replicate our results or extend MUBen to other datasets, backbone models, or uncertainty quantification (UQ) methods.
You can also find the example in this [Jupyter Notebook](https://github.com/Yinghao-Li/MUBen/blob/main/demo/demo.ipynb).

## Preparation

Currently, we do not provide Python wheels for installation as we aim to expose as many implementation details as possible.
You are recommended to directly use or modify the source code to conduct your experiments.
To do that, you can first `fork` the project and `clone` it to local with `git clone`.
```bash
git clone https://github.com/<your GitHub username>/MUBen.git
```
Or, you can directly clone this repository with `git` or GitHub CLI suppose you do not intend to do change tracking.
```bash
# clone with git
git clone https://github.com/Yinghao-Li/MUBen.git

# or, you can clone with GitHub CLI
gh repo clone Yinghao-Li/MUBen
```

The following operations assume that you are already in the project root directory `MUBen/`.

## Requirements

Our code is developed with `Python 3.10`.
Notice that it may **not** work with `Python < 3.9`.
It is recommended to create a new `conda` environment for this project.

```bash
conda create -n muben python=3.10
```

The required packages are listed in `requirements.txt`.
It is recommended to install these dependencies with `pip install` as `conda install` may sometimes encounter dependency resolution issue.
```bash
conda activate muben
pip install -r requirements.txt
```

### Docker
Alternatively, you can run this project in a docker container.
You can build your image through
```bash
docker build -t muben ./docker
```
and run your container in the terminal with
```bash
docker run --gpus all -it --rm  muben
```

### External dependencies

The backbone models `GROVER` and `Uni-Mol` require loading pre-trained model checkpoints.

- The `GROVER-base` checkpoint is available at GROVER's [project repo](https://github.com/tencent-ailab/grover) or can be directly downloaded through [this link](https://ai.tencent.com/ailab/ml/ml-data/grover-models/pretrain/grover_base.tar.gz).
Unzip the downloaded `.tar.gz` file to get the `.pt` checkpoint.
- The `Uni-Mol` checkpoint is available at Uni-Mol's [project repo](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol) or can be directly downloaded through [this link](https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt).

By default, the code will look for the models at locations `./models/grover_base.pt` and `./models/unimol_base.pt`, respectively.
You need to specify the `--checkpoint_path` argument if you prefer other locations and checkpoint names.

## A simple example

In this demonstration, we'll guide you through foundational training and testing of MUBen using the BBBP dataset as a minimal example.
We've chosen the DNN as our backbone model because it is both efficient and offers satisfactory performance.
For uncertainty quantification (UQ), we'll evaluate both the Deterministic method (referred to as "none" within MUBen) and Temperature Scaling.
While the procedures for other backbone models, UQ methods, or datasets are largely similar, you can explore specific variations by referring to API documentation.

### Importing packages
The first step is to import all the necessary packages from the MUBen source code that defines the datasets, backbone models, UQ methods, and trainers.

```python
import logging
import wandb
from transformers import set_seed

from muben.utils.selectors import configure_selector, dataset_selector, model_selector
from muben.train import Trainer
from muben.utils.io import set_logging

# initialize logger
logger = logging.getLogger(__name__)
```

### Deterministic method -- training

We first train the DNN model Deterministically.
That is, we do not apply any UQ method to the model.
Instead, we directly use its post-output-activation probabilities as its estimated reliability to its prediction.

Here we pass necessary hyper-parameters to the configuration to control the training process.
```python
# Set up the logging format and random seed.
# We do not use wandb for this demo, so we set its mode to "disabled".
set_logging()
set_seed(42)
wandb.init(mode="disabled",)

# Select the classes based on the descriptor type.
# DNN uses RDKit features, so we set the descriptor type to RDKit and select configuration, dataset,
# and model classes according to it.
descriptor_type = "RDKit"
config_class = configure_selector(descriptor_type)
dataset_class = dataset_selector(descriptor_type)
model_class = model_selector(descriptor_type)

# Specify the configuration of the experiment.
# Notice that although we directly edit the config object here, a more appropriate way of doing this is 
# passing arguments through the shell or json scripts when we are running the experiments through the terminal.
config = config_class()
config.model_name = "DNN"
config.feature_type = "rdkit"
config.data_folder = "../data/files/"
config.dataset_name = "bbbp"
config.result_folder = "../output-demo/"
config.uncertainty_method = "none"  # here "none" refers to "Deterministic"
config.retrain_model = True

# We only train the model for a few epochs for the demo.
config.n_epochs = 50
# activate training timer
config.time_training = True

# Post initialization of the arguments.
config.__post_init__()

# Load dataset metadata, validate the arguments, and log the configuration.
_ = config.get_meta().validate().log()
```
The configuration details are printed out in your terminal by calling `config.log()`.


Similar to configuration, we automatically infer the dataset and collate function classes according to the descriptor type we set above.
Then, we initialize the training, validation, and test datasets.
```python
# Load and process the training, validation and test datasets
dataset_class, collator_class = dataset_selector(descriptor_type)
training_dataset = dataset_class().prepare(config=config, partition="train")
valid_dataset = dataset_class().prepare(config=config, partition="valid")
test_dataset = dataset_class().prepare(config=config, partition="test")
```

Afterward, we can initialize the trainer and model with our configuration.
`model_selector` automatically detects the model type according to the descriptor.
In this case, [`DNN`](https://github.com/Yinghao-Li/MUBen/blob/0972667c69a3543ce0f6c3ce7689407d97dac153/muben/model/dnn/dnn.py#L17) is the selected model.
Then, the trainer initializes the model with arguments defined in the configuration.

```python
# Inintialized the trainer with the configuration and datasets
# Inintialized the trainer with the configuration and datasets
trainer = Trainer(
    config=config,
    model_class=model_selector(descriptor_type),
    training_dataset=training_dataset,
    valid_dataset=valid_dataset,
    test_dataset=test_dataset,
    collate_fn=collator_class(config),
).initialize(config=config)
```

Once the trainer is initialized, we can call `trainer.run()` to do the training.

```python
# Run the training, validation and test process.
# The model checkpoint and predicted results will be automatically saved in the specified output folder.
trainer.run()
```

### Temperature Scaling -- training

Training the DNN model with Temperature Scaling is basically identical to training with the Deterministic method.
The only difference is that we need to define the `uncertainty_method` in `config` as `"TemperatureScaling"` instead of `"none"`.

```python
wandb.init(mode="disabled",)
# Change some configuration items.
config.uncertainty_method = "TemperatureScaling"
config.retrain_model = False
config.n_ts_epochs = 10  # number of epochs for training the temperature scaling layer.
config.__post_init__()
_ = config.validate().log()

# Re-inintialized the trainer with the updated configuration.
# The datasets are not changed.
trainer = Trainer(
    config=config,
    model_class=model_selector(descriptor_type),
    training_dataset=training_dataset,
    valid_dataset=valid_dataset,
    test_dataset=test_dataset,
    collate_fn=collator_class(config),
).initialize(config=config)

# Run the training, validation and test process.
# The trainer will load the model checkpoint from the Deterministic run and
# continue training the temperature scaling layer.
# Notice that not all UQ methods support continued training. For example, BBP requires training from scratch.
trainer.run()
```


### Evaluation
Here, we provide a simplified version of metric calculation.
Please check `<project root>/assist/result_get_metrics.py` for the full function.

```python
import os.path as osp
import pandas as pd
from muben.utils.metrics import classification_metrics
from muben.utils.io import load_results


# Define the path to the predicted results. "det" stands for "Deterministic"; "ts" stands for "Temperature Scaling".
det_result = osp.join(
    config.result_folder, config.dataset_name, f"{config.model_name}-{config.feature_type}",
    "none", f"seed-{config.seed}", "preds", "0.pt"
)
ts_result = osp.join(
    config.result_folder, config.dataset_name, f"{config.model_name}-{config.feature_type}",
    "TemperatureScaling", f"seed-{config.seed}", "preds", "0.pt"
)

# Load the predicted results.
det_preds, _, lbs, masks = load_results([det_result])
ts_preds, _, _, _ = load_results([ts_result])

# Calculate the metrics.
det_metrics = classification_metrics(det_preds, lbs, masks)
ts_metrics = classification_metrics(ts_preds, lbs, masks)

det_metrics = {k: v['macro-avg'] for k, v in det_metrics.items()}
ts_metrics = {k: v['macro-avg'] for k, v in ts_metrics.items()}

# Present the results in a dataframe.
det_metrics_df = pd.DataFrame({"Deterministic": det_metrics, "TemperatureScaling": ts_metrics})
print(det_metrics_df.T)
```

The result will be presented as a table the columns being metrics and rows being the UQ method.