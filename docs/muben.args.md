<!-- markdownlint-disable -->


# <kbd>module</kbd> `muben.args`

Base classes for arguments and configurations. 

This module defines base classes for handling arguments and configurations across the application. It includes classes for model descriptor arguments, general arguments, and configurations that encompasses dataset, model, and training settings. 


---

## <kbd>class</kbd> `DescriptorArguments`
Model type arguments. 

This class holds the arguments related to the descriptor type of the model. It allows for specifying the type of descriptor used in model construction, with options including RDKit, Linear, 2D, and 3D descriptors. 



**Attributes:**
 
 - <b>`descriptor_type`</b> (str):  Descriptor type. Choices are ["RDKit", "Linear", "2D", "3D"]. 


---

## <kbd>class</kbd> `Arguments`
Base class for managing arguments related to model training, evaluation, and data handling. 



**Attributes:**
 
 - <b>`wandb_api_key`</b> (str):  The API key for Weights & Biases. Default is None. 
 - <b>`wandb_project`</b> (str):  The project name on Weights & Biases. Default is None. 
 - <b>`wandb_name`</b> (str):  The name of the model on Weights & Biases. Default is None. 
 - <b>`disable_wandb`</b> (bool):  Disable integration with Weights & Biases. Default is False. 
 - <b>`dataset_name`</b> (str):  Name of the dataset. Default is an empty string. 
 - <b>`data_folder`</b> (str):  Folder containing all datasets. Default is an empty string. 
 - <b>`data_seed`</b> (int):  Seed used for random data splitting. Default is None. 
 - <b>`result_folder`</b> (str):  Directory to save model outputs. Default is "./output". 
 - <b>`ignore_preprocessed_dataset`</b> (bool):  Whether to ignore pre-processed datasets. Default is False. 
 - <b>`disable_dataset_saving`</b> (bool):  Disable saving of pre-processed datasets. Default is False. 
 - <b>`disable_result_saving`</b> (bool):  Disable saving of training results and model checkpoints. Default is False. 
 - <b>`overwrite_results`</b> (bool):  Whether to overwrite existing outputs. Default is False. 
 - <b>`log_path`</b> (str):  Path for the logging file. Set to `disabled` to disable log saving. Default is None. 
 - <b>`descriptor_type`</b> (str):  Descriptor type. Choices are ["RDKit", "Linear", "2D", "3D"]. Default is None. 
 - <b>`model_name`</b> (str):  Name of the model. Default is "DNN". Choices are defined in MODEL_NAMES. 
 - <b>`dropout`</b> (float):  Dropout ratio. Default is 0.1. 
 - <b>`binary_classification_with_softmax`</b> (bool):  Use softmax for binary classification. Deprecated. Default is False. 
 - <b>`regression_with_variance`</b> (bool):  Use two output heads for regression (mean and variance). Default is False. 
 - <b>`retrain_model`</b> (bool):  Train model from scratch regardless of existing saved models. Default is False. 
 - <b>`ignore_uncertainty_output`</b> (bool):  Ignore saved uncertainty models/results. Load no-uncertainty model if possible. Default is False. 
 - <b>`ignore_no_uncertainty_output`</b> (bool):  Ignore checkpoints from no-uncertainty training processes. Default is False. 
 - <b>`batch_size`</b> (int):  Batch size for training. Default is 32. 
 - <b>`batch_size_inference`</b> (int):  Batch size for inference. Default is None. 
 - <b>`n_epochs`</b> (int):  Number of training epochs. Default is 50. 
 - <b>`lr`</b> (float):  Learning rate. Default is 1e-4. 
 - <b>`grad_norm`</b> (float):  Gradient norm for clipping. 0 means no clipping. Default is 0. 
 - <b>`lr_scheduler_type`</b> (str):  Type of learning rate scheduler. Default is "constant".  Choices include ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]. 
 - <b>`warmup_ratio`</b> (float):  Warm-up ratio for learning rate scheduler. Default is 0.1. 
 - <b>`seed`</b> (int):  Random seed for initialization. Default is 0. 
 - <b>`debug`</b> (bool):  Enable debugging mode with fewer data. Default is False. 
 - <b>`deploy`</b> (bool):  Enable deploy mode, avoiding runtime errors on bugs. Default is False. 
 - <b>`time_training`</b> (bool):  Measure training time per training step. Default is False. 
 - <b>`freeze_backbone`</b> (bool):  Freeze the backbone model during training. Only update the output layers. Default is False. 
 - <b>`valid_epoch_interval`</b> (int):  Interval of training epochs between each validation step. Set to 0 to disable validation. Default is 1. 
 - <b>`valid_tolerance`</b> (int):  Maximum allowed validation steps without performance increase. Default is 20. 
 - <b>`n_test`</b> (int):  Number of test loops in one training process. Default is 1. For some Bayesian methods, default is 20. 
 - <b>`test_on_training_data`</b> (bool):  Include test results on training data. Default is False. 
 - <b>`uncertainty_method`</b> (str):  Method for uncertainty estimation. Default is UncertaintyMethods.none. Choices are defined in UncertaintyMethods. 
 - <b>`n_ensembles`</b> (int):  Number of ensemble models in deep ensembles method. Default is 5. 
 - <b>`swa_lr_decay`</b> (float):  Learning rate decay coefficient during SWA training. Default is 0.5. 
 - <b>`n_swa_epochs`</b> (int):  Number of SWA training epochs. Default is 20. 
 - <b>`k_swa_checkpoints`</b> (int):  Number of SWA checkpoints for Gaussian covariance matrix. Should not exceed `n_swa_epochs`. Default is 20. 
 - <b>`ts_lr`</b> (float):  Learning rate for training temperature scaling parameters. Default is 0.01. 
 - <b>`n_ts_epochs`</b> (int):  Number of Temperature Scaling training epochs. Default is 20. 
 - <b>`apply_temperature_scaling_after_focal_loss`</b> (bool):  Apply temperature scaling after training with focal loss. Default is False. 
 - <b>`bbp_prior_sigma`</b> (float):  Sigma value for Bayesian Backpropagation prior. Default is 0.1. 
 - <b>`apply_preconditioned_sgld`</b> (bool):  Apply pre-conditioned Stochastic Gradient Langevin Dynamics instead of vanilla. Default is False. 
 - <b>`sgld_prior_sigma`</b> (float):  Variance of the SGLD Gaussian prior. Default is 0.1. 
 - <b>`n_langevin_samples`</b> (int):  Number of model checkpoints sampled from Langevin Dynamics. Default is 30. 
 - <b>`sgld_sampling_interval`</b> (int):  Number of epochs per SGLD sampling operation. Default is 2. 
 - <b>`evidential_reg_loss_weight`</b> (float):  Weight of evidential loss. Default is 1. 
 - <b>`evidential_clx_loss_annealing_epochs`</b> (int):  Epochs before evidential loss weight increases to 1. Default is 10. 
 - <b>`no_cuda`</b> (bool):  Disable CUDA even when available. Default is False. 
 - <b>`no_mps`</b> (bool):  Disable Metal Performance Shaders (MPS) even when available. Default is False. 
 - <b>`num_workers`</b> (int):  Number of threads for processing the dataset. Default is 0. 
 - <b>`num_preprocess_workers`</b> (int):  Number of threads for preprocessing the dataset. Default is 8. 
 - <b>`pin_memory`</b> (bool):  Pin memory for data loader for faster data transfer to CUDA devices. Default is False. 
 - <b>`n_feature_generating_threads`</b> (int):  Number of threads for generating features. Default is 8. 
 - <b>`enable_active_learning`</b> (bool):  Enable active learning. Default is False. 
 - <b>`n_init_instances`</b> (int):  Number of initial instances for active learning. Default is 100. 
 - <b>`n_al_select`</b> (int):  Number of instances to select in each active learning epoch. Default is 50. 
 - <b>`n_al_loops`</b> (int):  Number of active learning loops. Default is 5. 
 - <b>`al_random_sampling`</b> (bool):  Select instances randomly in active learning. Default is False. 



**Note:**

> This class contains many attributes. Each attribute controls a specific aspect of the training or evaluation process, including but not limited to data handling, model selection, training configurations, and evaluation metrics. 





---

## <kbd>class</kbd> `Config`
Extended configuration class inheriting from Arguments to include dataset-specific arguments. 

Inherits:  Arguments: Inherits all attributes from Arguments for comprehensive configuration management. 



**Attributes:**
 
 - <b>`classes`</b> (List[str]):  All possible classification classes. Default is None. 
 - <b>`task_type`</b> (str):  Type of task, e.g., "classification" or "regression". Default is "classification". 
 - <b>`n_tasks`</b> (int):  Number of tasks (sets of labels to predict). Default is None. 
 - <b>`eval_metric`</b> (str):  Metric for evaluating validation and test performance. Default is None. 
 - <b>`random_split`</b> (bool):  Whether the dataset is split randomly. Default is False. 



**Note:**

> The attributes defined in Config are meant to be overridden by dataset-specific metadata when used. 




---


### <kbd>function</kbd> `from_args`

```python
from_args(args)
```

Initialize configuration from an Arguments instance. 

This method updates the current configuration based on the values provided in an instance of the Arguments class or any subclass thereof. It's useful for transferring settings from command-line arguments or other configurations directly into this Config instance. 



**Args:**
 
 - <b>`args`</b>:  An instance of Arguments or a subclass containing configuration settings to be applied. 



**Returns:**
 
 - <b>`Config`</b>:  The instance itself, updated with the settings from `args`. 



**Note:**

> This method iterates through all attributes of `args` and attempts to set corresponding attributes in the Config instance. Attributes not present in Config will be ignored. 

---


### <kbd>function</kbd> `get_meta`

```python
get_meta(meta_dir: str = None, meta_file_name: str = 'meta.json')
```

Load meta file and update class attributes accordingly. 



**Args:**
 
 - <b>`meta_dir`</b> (str):  Directory containing the meta file. If not specified, uses `data_dir` attribute. 
 - <b>`meta_file_name`</b> (str):  Name of the meta file to load. Default is "meta.json". 



**Returns:**
 
 - <b>`Config`</b>:  The instance itself after updating attributes based on the meta file. 

---


### <kbd>function</kbd> `load`

```python
load(file_dir: str, file_name: str = 'config')
```

Load configuration from a JSON file. 



**Args:**
 
 - <b>`file_dir`</b> (str):  The directory where the configuration file is located. 
 - <b>`file_name`</b> (str):  The name of the file (without the extension) from which to load the configuration. Defaults to "config". 



**Raises:**
 
 - <b>`FileNotFoundError`</b>:  If the specified file does not exist or the directory does not contain the configuration file. 

---


### <kbd>function</kbd> `log`

```python
log()
```

Log the current configuration settings. 

Outputs the configuration settings to the logging system, formatted for easy reading. 

---


### <kbd>function</kbd> `save`

```python
save(file_dir: str, file_name: str = 'config')
```

Save the current configuration to a JSON file. 



**Args:**
 
 - <b>`file_dir`</b> (str):  The directory where the configuration file will be saved. 
 - <b>`file_name`</b> (str):  The name of the file (without the extension) to save the configuration. Defaults to "config". 



**Raises:**
 
 - <b>`FileNotFoundError`</b>:  If the specified directory does not exist. 
 - <b>`Exception`</b>:  If there is an issue saving the file. 

---


### <kbd>function</kbd> `validate`

```python
validate()
```

Validate the configuration. 

Checks for argument conflicts and resolves them if possible, issuing warnings for any discrepancies found. Ensures that the model name, feature type, and uncertainty methods are compatible with the specified task type. 



**Raises:**
 
 - <b>`AssertionError`</b>:  If an incompatible configuration is detected that cannot be automatically resolved. 

