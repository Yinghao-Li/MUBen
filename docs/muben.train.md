<!-- markdownlint-disable -->


# <kbd>module</kbd> `muben.train`

Base trainer functions to facilitate training, validation, and testing  of machine learning models. This Trainer class is designed to seamlessly  integrate with various datasets, loss functions, metrics, and uncertainty  estimation methods. It provides convenient mechanisms to standardize,  initialize and manage training states, and is also integrated with logging  and Weights & Biases (wandb) for experiment tracking. 


---



## <kbd>class</kbd> `Trainer`
This Trainer class is designed to facilitate the training, validation, and testing of machine learning models. It integrates with various datasets, loss functions, metrics, and uncertainty estimation methods, providing mechanisms to standardize, initialize, and manage training states. It supports logging and integration with Weights & Biases (wandb) for experiment tracking. 



### <kbd>method</kbd> `__init__`

```python
__init__(
    config,
    model_class=None,
    training_dataset=None,
    valid_dataset=None,
    test_dataset=None,
    collate_fn=None,
    scalar=None,
    **kwargs
)
```

Initializes the Trainer object. 



**Args:**
 
 - <b>`config`</b> (Config):  Configuration object containing all necessary parameters for training. 
 - <b>`model_class`</b> (optional):  The class of the model to be trained. 
 - <b>`training_dataset`</b> (Dataset, optional):  Dataset for training the model. 
 - <b>`valid_dataset`</b> (Dataset, optional):  Dataset for validating the model. 
 - <b>`test_dataset`</b> (Dataset, optional):  Dataset for testing the model. 
 - <b>`collate_fn`</b> (Callable, optional):  Function to collate data samples into batches. 
 - <b>`scalar`</b> (StandardScaler, optional):  Scaler for standardizing input data. 
 - <b>`**kwargs`</b>:  Additional keyword arguments for configuration adjustments. 


---

### <kbd>property</kbd> backbone_params

Retrieves parameters of the model's backbone, excluding the output layer. 

Useful for operations that need to differentiate between backbone and output layer parameters, such as freezing the backbone during training. 

**Returns:**
 
 - <b>`list`</b>:  Parameters of the model's backbone. 

### <kbd>property</kbd> config

Retrieves the configuration of the Trainer. 

### <kbd>property</kbd> model

Retrieves the scaled model if available, otherwise returns the base model. 

### <kbd>property</kbd> n_model_parameters

Computes the total number of trainable parameters in the model. 

### <kbd>property</kbd> n_training_steps

The number of total training steps 

### <kbd>property</kbd> n_update_steps_per_epoch

Calculates the number of update steps required per epoch. 


**Returns:**
 
 - <b>`int`</b>:  Number of update steps per epoch. 

### <kbd>property</kbd> n_valid_steps

The number of total validation steps 

### <kbd>property</kbd> test_dataset


### <kbd>property</kbd> training_dataset



### <kbd>property</kbd> valid_dataset

---


### <kbd>method</kbd> `eval_and_save`

```python
eval_and_save()
```

Evaluates the model's performance on the validation dataset and saves it if its performance is improved. 

This method is part of the training loop where the model is periodically evaluated on the validation dataset, and the best-performing model state is saved. 

---



### <kbd>method</kbd> `evaluate`

```python
evaluate(
    dataset,
    n_run: Optional[int] = 1,
    return_preds: Optional[bool] = False
)
```

Evaluates the model's performance on the given dataset. 



**Args:**
 
 - <b>`dataset`</b> (Dataset):  The dataset to evaluate the model on. 
 - <b>`n_run`</b> (int, optional):  Number of runs for evaluation. Defaults to 1. 
 - <b>`return_preds`</b> (bool, optional):  Whether to return the predictions along with metrics. Defaults to False. 



**Returns:**
 
 - <b>`dict, or (dict, numpy.ndarray or Tuple[numpy.ndarray, numpy.ndarray])`</b>:  Evaluation metrics, or tuple containing metrics and predictions based on `return_preds`. 

---



### <kbd>method</kbd> `freeze`

```python
freeze()
```

Freezes all model parameters, preventing them from being updated during training. 



**Returns:**
 
 - <b>`Trainer`</b>:  The current instance with model parameters frozen. 

---



### <kbd>method</kbd> `freeze_backbone`

```python
freeze_backbone()
```

Freezes the backbone parameters of the model, preventing them from being updated during training. 



**Returns:**
 
 - <b>`Trainer`</b>:  The current instance with backbone parameters frozen. 

---



### <kbd>method</kbd> `get_dataloader`

```python
get_dataloader(
    dataset,
    shuffle: Optional[bool] = False,
    batch_size: Optional[int] = 0
)
```

Creates a DataLoader for the specified dataset. 



**Args:**
 
 - <b>`dataset`</b>:  Dataset for which the DataLoader is to be created. 
 - <b>`shuffle`</b> (bool, optional):  Whether to shuffle the data. Defaults to False. 
 - <b>`batch_size`</b> (int, optional):  Batch size for the DataLoader. Uses the batch size from the configuration if not specified. 



**Returns:**
 
 - <b>`DataLoader`</b>:  The created DataLoader for the provided dataset. 

---



### <kbd>method</kbd> `get_loss`

```python
get_loss(logits, batch, n_steps_per_epoch=None) → Tensor
```

Computes the loss for a batch of data. 

This method can be overridden by subclasses to implement custom loss computation logic. 



**Args:**
 
 - <b>`logits`</b> (torch.Tensor):  The predictions or logits produced by the model for the given batch. 
 - <b>`batch`</b> (Batch):  The batch of training data. 
 - <b>`n_steps_per_epoch`</b> (int, optional):  Represents the number of batches in a training epoch,  used specifically for certain uncertainty methods like Bayesian Backpropagation (BBP). 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  The computed loss for the batch. 

---



### <kbd>method</kbd> `get_metrics`

```python
get_metrics(lbs, preds, masks)
```

Calculates evaluation metrics based on the given labels, predictions, and masks. 

This method computes the appropriate metrics based on the task type (classification or regression). 



**Args:**
 
 - <b>`lbs`</b> (numpy.ndarray):  Ground truth labels. 
 - <b>`preds`</b> (numpy.ndarray):  Model predictions. 
 - <b>`masks`</b> (numpy.ndarray):  Masks indicating valid entries in labels and predictions. 



**Returns:**
 
 - <b>`dict`</b>:  Computed metrics for evaluation. 

---



### <kbd>method</kbd> `inference`

```python
inference(dataset, **kwargs)
```

Conducts inference over an entire dataset using the model. 



**Args:**
 
 - <b>`dataset`</b> (Dataset):  The dataset for which inference needs to be performed. 
 - <b>`**kwargs`</b>:  Additional keyword arguments. 



**Returns:**
 
 - <b>`numpy.ndarray`</b>:  The model outputs as logits or a tuple of logits. 

---



### <kbd>method</kbd> `initialize`

```python
initialize(*args, **kwargs)
```

Initializes the trainer's status and its key components including the model, optimizer, learning rate scheduler, and loss function. 

This method sets up the training environment by initializing the model, optimizer, learning rate scheduler, and the loss function based on the provided configuration. It also prepares the trainer for logging and checkpointing mechanisms. 



**Args:**
 
 - <b>`*args`</b>:  Variable length argument list for model initialization. 
 - <b>`**kwargs`</b>:  Arbitrary keyword arguments for model initialization. 



**Returns:**
 
 - <b>`Trainer`</b>:  The initialized Trainer instance ready for training. 

---



### <kbd>method</kbd> `initialize_loss`

```python
initialize_loss(disable_focal_loss=False)
```

Initializes the loss function based on the task type and specified uncertainty method. 

This method sets up the appropriate loss function for the training process, considering the task type (classification or regression) and whether any specific uncertainty methods (e.g., evidential or focal loss) are applied. 



**Args:**
 
 - <b>`disable_focal_loss`</b> (bool, optional):  If True, disables the use of focal loss, even if  specified by the uncertainty method. Defaults to False. 



**Returns:**
 
 - <b>`Trainer`</b>:  The Trainer instance with the initialized loss function. 

---



### <kbd>method</kbd> `initialize_model`

```python
initialize_model(*args, **kwargs)
```

Abstract method to initialize the model. 

This method should be implemented in subclasses of Trainer, providing the specific logic to initialize the model that will be used for training. 



**Returns:**
 
 - <b>`Trainer`</b>:  The Trainer instance with the model initialized. 

---



### <kbd>method</kbd> `initialize_optimizer`

```python
initialize_optimizer(*args, **kwargs)
```

Initializes the model's optimizer based on the set configurations. 

This method sets up the optimizer for the model's parameters. It includes special handling for SGLD-based uncertainty methods by differentiating between backbone and output layer parameters. 



**Args:**
 
 - <b>`*args`</b>:  Variable length argument list for optimizer initialization. 
 - <b>`**kwargs`</b>:  Arbitrary keyword arguments for optimizer initialization. 



**Returns:**
 
 - <b>`Trainer`</b>:  The Trainer instance with the initialized optimizer. 

---



### <kbd>method</kbd> `initialize_scheduler`

```python
initialize_scheduler()
```

Initializes the learning rate scheduler based on the training configuration. 

This method sets up the learning rate scheduler using the total number of training steps and the specified warmup ratio. 



**Returns:**
 
 - <b>`Trainer`</b>:  The Trainer instance with the initialized scheduler. 

---



### <kbd>method</kbd> `inverse_standardize_preds`

```python
inverse_standardize_preds(
    preds: Union[ndarray, Tuple[ndarray, ndarray]]
) → Union[ndarray, Tuple[ndarray, ndarray]]
```

Transforms predictions back to their original scale if they have been standardized. 



**Args:**
 
 - <b>`preds`</b> (numpy.ndarray or Tuple[numpy.ndarray, numpy.ndarray]):  Model predictions, can either be a single  array or a tuple containing two arrays for mean and variance, respectively. 



**Returns:**
 
 - <b>`numpy.ndarray or Tuple[numpy.ndarray, numpy.ndarray]`</b>:  Inverse-standardized predictions. 

---



### <kbd>method</kbd> `load_checkpoint`

```python
load_checkpoint()
```

Loads the model from a checkpoint. 

This method attempts to load the model checkpoint from the configured path. It supports loading with and without considering the uncertainty estimation method used during training. 



**Returns:**
 
 - <b>`bool`</b>:  True if the model is successfully loaded from a checkpoint, otherwise False. 

---



### <kbd>method</kbd> `log_results`

```python
log_results(
    metrics: dict,
    logging_func=<bound method Logger.info of <Logger trainer.trainer (WARNING)>>
)
```

Logs evaluation metrics using the specified logging function. 



**Args:**
 
 - <b>`metrics`</b> (dict):  Dictionary containing evaluation metrics to be logged. 
 - <b>`logging_func`</b> (function, optional):  Logging function to which metrics will be sent. Defaults to `logger.info`. 



**Returns:**
 None 

---



### <kbd>method</kbd> `process_logits`

```python
process_logits(logits: ndarray) → Union[ndarray, Tuple[ndarray, ndarray]]
```

Processes the output logits based on the training tasks or variants. 



**Args:**
 
 - <b>`logits`</b> (numpy.ndarray):  The raw logits produced by the model. 



**Returns:**
 
 - <b>`numpy.ndarray or Tuple[numpy.ndarray, numpy.ndarray]`</b>:  Processed logits or a tuple containing processed logits based on the task type. 

---



### <kbd>method</kbd> `run`

```python
run()
```

Executes the training and evaluation process. 

This method serves as the main entry point for the training process, orchestrating the execution based on the specified uncertainty method. It handles different training strategies like ensembles, SWAG, temperature scaling, and more. 



**Returns:**
  None 

---



### <kbd>method</kbd> `run_ensembles`

```python
run_ensembles()
```

Trains and evaluates an ensemble of models. 

This method is used for uncertainty estimation through model ensembles, training multiple models with different seeds and evaluating their collective performance. 



**Returns:**
 
 - <b>`Trainer`</b>:  Self reference to the Trainer object, allowing for method chaining. 

---



### <kbd>method</kbd> `run_focal_loss`

```python
run_focal_loss()
```

Runs the training and evaluation pipeline utilizing focal loss. 

Focal loss is used to address class imbalance by focusing more on hard-to-classify examples. Temperature scaling can optionally be applied after training with focal loss. 



**Returns:**
 
 - <b>`Trainer`</b>:  Self reference to the Trainer object, allowing for method chaining. 

---



### <kbd>method</kbd> `run_iso_calibration`

```python
run_iso_calibration()
```

Performs isotonic calibration. 

Isotonic calibration is applied to calibrate the uncertainties of the model's predictions, based on the approach described in 'Accurate Uncertainties for Deep Learning Using Calibrated Regression'. 



**Returns:**
 
 - <b>`Trainer`</b>:  Self reference to the Trainer object, allowing for method chaining. 

---



### <kbd>method</kbd> `run_sgld`

```python
run_sgld()
```

Executes training and evaluation with Stochastic Gradient Langevin Dynamics (SGLD). 

SGLD is used for uncertainty estimation, incorporating random noise into the gradients to explore the model's parameter space more broadly. 



**Returns:**
 
 - <b>`Trainer`</b>:  Self reference to the Trainer object, allowing for method chaining. 

---



### <kbd>method</kbd> `run_single_shot`

```python
run_single_shot(apply_test=True)
```

Runs the training and evaluation pipeline for a single iteration. 

This method handles the process of training the model and optionally evaluating it on a test dataset. It is designed for a straightforward, single iteration of training and testing. 



**Args:**
 
 - <b>`apply_test`</b> (bool, optional):  Whether to run the test function as part of the process. Defaults to True. 



**Returns:**
 
 - <b>`Trainer`</b>:  Self reference to the Trainer object, allowing for method chaining. 

---



### <kbd>method</kbd> `run_swag`

```python
run_swag()
```

Executes the training and evaluation pipeline using the SWAG method. 

Stochastic Weight Averaging Gaussian (SWAG) is used for uncertainty estimation. This method involves training the model with early stopping and applying SWAG for post-training uncertainty estimation. 



**Returns:**
 
 - <b>`Trainer`</b>:  Self reference to the Trainer object, allowing for method chaining. 

---



### <kbd>method</kbd> `run_temperature_scaling`

```python
run_temperature_scaling()
```

Executes the training and evaluation pipeline with temperature scaling. 

Temperature scaling is applied as a post-processing step to calibrate the confidence of the model's predictions. 



**Returns:**
 
 - <b>`Trainer`</b>:  Self reference to the Trainer object, allowing for method chaining. 

---



### <kbd>method</kbd> `save_checkpoint`

```python
save_checkpoint()
```

Saves the current model state as a checkpoint. 

This method checks the `disable_result_saving` configuration flag before saving. If saving is disabled, it logs a warning and does not perform the save operation. 



**Returns:**
 
 - <b>`Trainer`</b>:  The current instance after attempting to save the model checkpoint. 

---



### <kbd>method</kbd> `save_results`

```python
save_results(path, preds, variances, lbs, masks)
```

Saves the model predictions, variances, ground truth labels, and masks to disk. 

This method saves the results of model predictions to a specified path. It is capable of handling both the predictions and their associated variances, along with the ground truth labels and masks that indicate which data points should be considered in the analysis. If the configuration flag `disable_result_saving` is set to True, the method will log a warning and not perform any saving operation. 



**Args:**
 
 - <b>`path`</b> (str):  The destination path where the results will be saved. 
 - <b>`preds`</b> (array_like):  The predictions generated by the model. 
 - <b>`variances`</b> (array_like):  The variances associated with each prediction, indicating the uncertainty of the predictions. 
 - <b>`lbs`</b> (array_like):  The ground truth labels against which the model's predictions can be evaluated. 
 - <b>`masks`</b> (array_like):  Masks indicating which data points are valid and should be considered in the evaluation. 



**Returns:**
 
 - <b>`None`</b>:  This method does not return any value. 

---



### <kbd>method</kbd> `set_mode`

```python
set_mode(mode: str)
```

Sets the training mode for the model. 

Depending on the mode, the model is set to training, evaluation, or testing. This method is essential for correctly configuring the model's state for different phases of the training and evaluation process. 



**Args:**
 
 - <b>`mode`</b> (str):  The mode to set the model to. Should be one of 'train', 'eval', or 'test'. 



**Returns:**
 
 - <b>`Trainer`</b>:  The Trainer instance with the model set to the specified mode. 

---



### <kbd>method</kbd> `standardize_training_lbs`

```python
standardize_training_lbs()
```

Standardizes the label distribution of the training dataset for regression tasks. 

This method applies standardization to the labels of the training dataset, transforming them to a standard Gaussian distribution. It's applicable only for regression tasks. 



**Returns:**
 
 - <b>`Trainer`</b>:  The Trainer instance with standardized training labels. 

---



### <kbd>method</kbd> `swa_session`

```python
swa_session()
```

Executes the SWA session. 

This method is intended to be overridden by child classes for specialized handling of optimizer or model initialization required by SWA (Stochastic Weight Averaging). 



**Returns:**
 
 - <b>`Trainer`</b>:  Self reference to the Trainer object, allowing for method chaining. 

---



### <kbd>method</kbd> `test`

```python
test(load_best_model=True, return_preds=False)
```

Tests the model's performance on the test dataset. 



**Args:**
 
 - <b>`load_best_model`</b> (bool, optional):  Whether to load the best model saved during training for testing. Defaults to True. 
 - <b>`return_preds`</b> (bool, optional):  Whether to return the predictions along with metrics. Defaults to False. 



**Returns:**
 
 - <b>`dict, or tuple[dict, numpy.ndarray or Tuple[numpy.ndarray, numpy.ndarray]]`</b>:  Evaluation metrics (and predictions) for the test dataset. 

---



### <kbd>method</kbd> `test_on_training_data`

```python
test_on_training_data(
    load_best_model=True,
    return_preds=False,
    disable_result_saving=False
)
```

Tests the model's performance on the training dataset. 

This method is useful for understanding the model's performance on the data it was trained on, which can provide insights into overfitting or underfitting. 



**Args:**
 
 - <b>`load_best_model`</b> (bool, optional):  If True, loads the best model saved during training. Defaults to True. 
 - <b>`return_preds`</b> (bool, optional):  If True, returns the predictions along with the evaluation metrics. Defaults to False. 
 - <b>`disable_result_saving`</b> (bool, optional):  If True, disables saving the results to disk. Defaults to False. 



**Returns:**
 
 - <b>`dict, or tuple[dict, numpy.ndarray or Tuple[numpy.ndarray, numpy.ndarray]]`</b>:  Evaluation metrics, or a tuple containing metrics and predictions if `return_preds` is True. 

---



### <kbd>method</kbd> `train`

```python
train(use_valid_dataset=False)
```

Executes the training process for the model. 

Optionally allows for training using the validation dataset instead of the training dataset. This option can be useful for certain model calibration techniques like temperature scaling. 



**Args:**
 
 - <b>`use_valid_dataset`</b> (bool, optional):  Determines if the validation dataset should be used  for training instead of the training dataset. Defaults to False. 



**Returns:**
 
 - <b>`None`</b>:  This method returns None. 

---



### <kbd>method</kbd> `training_epoch`

```python
training_epoch(data_loader)
```

Performs a single epoch of training using the provided data loader. 

This method iterates over the data loader, performs the forward pass, computes the loss, and updates the model parameters. 



**Args:**
 
 - <b>`data_loader`</b> (DataLoader):  DataLoader object providing batches of training data. 



**Returns:**
 
 - <b>`float`</b>:  The average training loss for the epoch. 

---



### <kbd>method</kbd> `ts_session`

```python
ts_session()
```

Executes the temperature scaling session. 

This session involves retraining the model on the validation set with a modified learning rate and epochs to apply temperature scaling for model calibration. 



**Returns:**
 
 - <b>`Trainer`</b>:  Self reference to the Trainer object, allowing for method chaining. 

---



### <kbd>method</kbd> `unfreeze`

```python
unfreeze()
```

Unfreezes all model parameters, allowing them to be updated during training. 



**Returns:**
 
 - <b>`Trainer`</b>:  The current instance with model parameters unfrozen. 

---



### <kbd>method</kbd> `unfreeze_backbone`

```python
unfreeze_backbone()
```

Unfreezes the backbone parameters of the model, allowing them to be updated during training. 



**Returns:**
 
 - <b>`Trainer`</b>:  The current instance with backbone parameters unfrozen. 

