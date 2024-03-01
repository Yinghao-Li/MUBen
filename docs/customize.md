# Customization

We have shown in the previous sections how to reproduce our results through a Python (Jupyter) script or command line interface.
In this section, we focus on extending our benchmark to more datasets and backbone models.

!!! Note
    We have also tried to modulate the UQ methods.
    However, it seems that most of their implementations are deeply entangled with the model architecture and training process.
    Therefore, we only provide a limited selection of UQ methods in MUBen and leave its extension to the community.


## Customize dataset

### Prepare the dataset

Training the backbone/UQ methods with a customized dataset is quite straightforward.
If you want to test the UQ methods on your own dataset, you can organize your data as `pandas.DataFrame` with three keys: `["smiles", "labels", "masks"]`.
Their types are shown below.
```
{
  "smiles": list of `str`,
  "labels": list of list of int/float,
  "masks": list of list of int/float (with values within {0,1})
}
```
Here, `mask=1` indicates the existence informative label at the position and `mask=0` indicates the missing label.

!!! Note
    You should store your molecules using smiles even if you choose other descriptors such as 2D and 3D graphs.
    The graphs or RDKit features could be constructed during data pre-processing within the training process.

The training, validation, and test partitions should be stored as `train.csv`, `valid.csv`, and `test.csv` files respectively in the same folder.
The `.csv` files should be accompanied by a `meta.json` file within the same directory.
It stores some constant dataset properties, *e.g.*, `task_type` (classification or regression), `n_tasks`, or `classes` (`[0,1]` for all our classification datasets).
For the customized dataset, one **required** property is the `eval_metric` for validation and testing (*e.g.*, roc-auc, rmse, *etc.*).

You can check the prepared datasets included in our program for reference.
You are recommended to put the dataset files in the `./data/file/<dataset name>` directory, but you can of course choose your favorite location and specify the `--data_folder` argument.

### Train the model

To conduct training and evaluation on the customized dataset, we only need to modify the `dataset_name` argument (`muben.args`) to the name of the customized dataset.
This can be achieved through both CLI (`--dataset_name <the name of your dataset>`) or within the Python script (`config.dataset_name="<the name of your dataset>"`).

!!! Note
    Notice that `dataset_name` only contains the name of the specific dataset folder instead of the entire path to a specific file.
    The full path to the training partition, for example, is constructed from `<dataset_folder>/<dataset_name>/train.csv`.


## Customize backbone model

It is also easy to define a customized backbone model and integrate it into the training & evaluation pipeline, as long as it follows the standard input & output format.
In the following example, we manually construct a DNN model that uses RDKit features as input.

### Define the model

The following code defines a conventional DNN model with customizable input/output/hidden dimensionalities and dropout probabilities.
For the output layer, we use the `OutputLayer` class defined in `muben.layers` module to realize easy initialization and integration of Bayes-by-Backprop (BBP).

```python
import torch.nn as nn

from muben.layers import OutputLayer


class DNN(nn.Module):
    def __init__(self,
                 d_feature: int,
                 n_lbs: int,
                 n_tasks: int,
                 hidden_dims: list[int],
                 p_dropout: float = 0.1,
                 uncertainty_method: str = "none",
                 **kwargs):
        super().__init__()

        # d_feature = config.d_feature
        # n_lbs = config.n_lbs
        # n_tasks = config.n_tasks
        # n_hidden_layers = config.n_dnn_hidden_layers
        # d_hidden = config.d_dnn_hidden
        # p_dropout = config.dropout
        # uncertainty_method = config.uncertainty_method

        if hidden_dims is None:
            hidden_dims = [d_hidden] * (n_hidden_layers + 1)
        else:
            n_hidden_layers = len(hidden_dims)

        self.input_layer = nn.Sequential(
            nn.Linear(d_feature, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(p_dropout),
        )

        hidden_layers = [
            nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(p_dropout),
            )
            for i in range(n_hidden_layers)
        ]
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.output_layer = OutputLayer(
            hidden_dims[-1], n_lbs * n_tasks, uncertainty_method, **kwargs
        )

        self.initialize()

    def initialize(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.apply(init_weights)
        self.output_layer.initialize()
        return self

    def forward(self, batch):
        features = batch.features

        x = self.input_layer(features)
        x = self.hidden_layers(x)

        logits = self.output_layer(x)

        return logits
```

### Initialize trainer with customized model

Once the model is defined, we can pass it as an argument to the `Trainer` class to set it as the backbone mode.
Notice that when the model is referred to but not initialized together with `Trainer`. 

For example, we can use the same code in the simple example for `Trainer` initialization.

```python
from muben.args import Config
from muben.utils.selectors import dataset_selector, model_selector
from muben.train import Trainer

descriptor_type = "RDKit"  # as mentioned above, we use RDKit features here
config_class = configure_selector(descriptor_type)
dataset_class = dataset_selector(descriptor_type)
config = Config()  # We'll can the default configuration for customized backbone models

# io configurations
config.feature_type = "rdkit"
config.data_folder = "./data/files/"
config.dataset_name = "bbbp"
config.result_folder = "./output-demo/"
config.uncertainty_method = "none"  # here "none" refers to "Deterministic"
# training configurations
config.retrain_model = True
config.n_epochs = 50
config.lr = 0.0001
# we'll leave model configurations
config.__post_init__()
config.get_meta().validate()

# Load and process the training, validation and test datasets
dataset_class, collator_class = dataset_selector(descriptor_type)
training_dataset = dataset_class().prepare(config=config, partition="train")
valid_dataset = dataset_class().prepare(config=config, partition="valid")
test_dataset = dataset_class().prepare(config=config, partition="test")

# Inintialized the trainer with the configuration and datasets
trainer = Trainer(
    config=config,
    model_class=DNN,  # passed the customized model class to the Trainer
    training_dataset=training_dataset,
    valid_dataset=valid_dataset,
    test_dataset=test_dataset,
    collate_fn=collator_class(config),
)
```

With the above code, we have initialized the trainer with `DNN` as well as RDKit datasets.
However, the model is not initialized.
To initialize the model, we can use `trainer.initialize`.

```python
trainer.initialize(
    d_feature = 200  # the feature dimensionality has to be the same as the output of your feature generator
    n_lbs = 1  # bbbp dataset has 2 label types (0, 1), but we use only 1 classification head for binary classification
    n_tasks = 1  # bbbp dataset has 1 tasks
    hidden_dims = [512, 512, 512]  # 3 hidden layer, each with dimensionality 512
    uncertainty_method = "MCDropout"  # set the uncertainty method as MC Dropout
)
```
The keyword arguments passed to `trainer.initialize` should be the same as what you have defined in `DNN.__init__` as we use `**kwargs` to pass these arguments.
If the keywords are different, the model may not be initialized properly.

Once the trainer has been initialized, we can start the training and evaluation process as we demonstrated before.
```python
trainer.run()
```
