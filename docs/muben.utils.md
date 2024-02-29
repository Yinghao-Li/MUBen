<!-- markdownlint-disable -->

# <kbd>module</kbd> `muben.utils`

Utility functions to support argument parsing, data loading, model training and evalution.

---

## <kbd>module</kbd> `argparser`

Argument Parser.

!!! Note
    The `ArgumentParser` class is modified from `huggingface/transformers` implementation.

---

### <kbd>class</kbd> `ArgumentParser`
This subclass of `argparse.ArgumentParser` uses type hints on dataclasses to generate arguments. 

The class is designed to play well with the native argparse. In particular, you can add more (non-dataclass backed) arguments to the parser after initialization and you'll get the output back after parsing as an additional namespace. Optional: To create sub argument groups use the `_argument_group_name` attribute in the dataclass. 



#### <kbd>method</kbd> `__init__`

```python
__init__(
    dataclass_types: Union[DataClassType, Iterable[DataClassType]],
    **kwargs
)
```



**Args:**
  dataclass_types:  Dataclass type, or list of dataclass types for which we will "fill" instances with the parsed args. 

**kwargs:**
  (Optional) Passed to `argparse.ArgumentParser()` in the regular way. 




---



#### <kbd>method</kbd> `parse_args_into_dataclasses`

```python
parse_args_into_dataclasses(
    args=None,
    return_remaining_strings=False,
    look_for_args_file=True,
    args_filename=None,
    args_file_flag=None
) → Tuple[DataClass, ]
```

Parse command-line args into instances of the specified dataclass types. 

This relies on argparse's `ArgumentParser.parse_known_args`. See the doc at: docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args 



**Args:**
 

**args:**
  List of strings to parse. The default is taken from sys.argv. (same as argparse.ArgumentParser) return_remaining_strings:  If true, also return a list of remaining argument strings. look_for_args_file:  If true, will look for a ".args" file with the same base name as the entry point script for this  process, and will append its potential content to the command line args. args_filename:  If not None, will uses this file instead of the ".args" file specified in the previous argument. args_file_flag:  If not None, will look for a file in the command-line args specified with this flag. The flag can be  specified multiple times and precedence is determined by the order (last one wins). 



**Returns:**
  Tuple consisting of: 

  - the dataclass instances in the same order as they were passed to the initializer.abspath 
  - if applicable, an additional namespace for more (non-dataclass backed) arguments added to the parser  after initialization. 
  - The potential list of remaining argument strings. (same as argparse.ArgumentParser.parse_known_args) 

---



#### <kbd>method</kbd> `parse_dict`

```python
parse_dict(args: Dict[str, Any]) → Tuple[DataClass, ]
```

Alternative helper method that does not use `argparse` at all, instead uses a dict and populating the dataclass types. 



**Args:**
  args (`dict`):  dict containing config values 



**Returns:**
  Tuple consisting of: 

  - the dataclass instances in the same order as they were passed to the initializer. 

---



#### <kbd>method</kbd> `parse_json_file`

```python
parse_json_file(json_file: str) → Tuple[DataClass, ]
```

Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the dataclass types. 



**Args:**
  json_file (`str` or `os.PathLike`):  File name of the json file to parse 



**Returns:**
  Tuple consisting of: 

  - the dataclass instances in the same order as they were passed to the initializer. 

---



#### <kbd>method</kbd> `parse_yaml_file`

```python
parse_yaml_file(yaml_file: str) → Tuple[DataClass, ]
```

Alternative helper method that does not use `argparse` at all, instead loading a yaml file and populating the dataclass types. 



**Args:**
  yaml_file (`str` or `os.PathLike`):  File name of the yaml file to parse 



**Returns:**
  Tuple consisting of: 

  - the dataclass instances in the same order as they were passed to the initializer. 

---


## <kbd>module</kbd> `chem`
Molecular descriptors and features 

---


### <kbd>function</kbd> `smiles_to_2d_coords`

```python
smiles_to_2d_coords(smiles)
```

Converts SMILES strings to 2D coordinates. 



**Args:**
 
 - <b>`smiles`</b> (str):  A SMILES string representing the molecule. 



**Returns:**
 
 - <b>`numpy.ndarray`</b>:  A 2D array of coordinates for the molecule. 


---


### <kbd>function</kbd> `smiles_to_3d_coords`

```python
smiles_to_3d_coords(smiles, n_conformer)
```

Converts SMILES strings to 3D coordinates. 



**Args:**
 
 - <b>`smiles`</b> (str):  A SMILES string representing the molecule. 
 - <b>`n_conformer`</b> (int):  Number of conformers to generate for the molecule. 



**Returns:**
 
 - <b>`list[numpy.ndarray]`</b>:  A list of 3D arrays of coordinates for each conformer of the molecule. 


---



### <kbd>function</kbd> `smiles_to_coords`

```python
smiles_to_coords(smiles, n_conformer=10)
```

Converts SMILES strings to 3D coordinates. 



**Args:**
 
 - <b>`smiles`</b> (str):  A SMILES string representing the molecule. 
 - <b>`n_conformer`</b> (int):  Number of conformers to generate for the molecule. 



**Returns:**
 
 - <b>`list[numpy.ndarray]`</b>:  A list of 3D arrays of coordinates for each conformer of the molecule. 


---



### <kbd>function</kbd> `rdkit_2d_features_normalized_generator`

```python
rdkit_2d_features_normalized_generator(mol) → ndarray
```

Generates RDKit 2D normalized features for a molecule. 



**Args:**
 
 - <b>`mol`</b> (str or Chem.Mol):  A molecule represented as a SMILES string or an RDKit molecule object. 



**Returns:**
 
 - <b>`numpy.ndarray`</b>:  An array containing the RDKit 2D normalized features. 


---



### <kbd>function</kbd> `morgan_binary_features_generator`

```python
morgan_binary_features_generator(
    mol,
    radius: int = 2,
    num_bits: int = 1024
) → ndarray
```

Generates a binary Morgan fingerprint for a molecule. 



**Args:**
 
 - <b>`mol`</b> (str or Chem.Mol):  A molecule represented as a SMILES string or an RDKit molecule object. 
 - <b>`radius`</b> (int):  Radius of the Morgan fingerprint. 
 - <b>`num_bits`</b> (int):  Number of bits in the Morgan fingerprint. 



**Returns:**
 
 - <b>`numpy.ndarray`</b>:  An array containing the binary Morgan fingerprint. 


---



### <kbd>function</kbd> `smiles_to_atom_ids`

```python
smiles_to_atom_ids(smiles: str) → list[int]
```

Converts SMILES strings to a list of atom IDs with hydrogens included. 



**Args:**
 
 - <b>`smiles`</b> (str):  A SMILES string representing the molecule. 



**Returns:**
 
 - <b>`list[int]`</b>:  A list of atomic numbers corresponding to the atoms in the molecule. 


---



### <kbd>function</kbd> `atom_to_atom_ids`

```python
atom_to_atom_ids(atoms: list[str]) → list[int]
```

Converts a list of atom symbols to a list of atom IDs. 



**Args:**
 
 - <b>`atoms`</b> (list[str]):  A list of atom symbols. 



**Returns:**
 
 - <b>`list[int]`</b>:  A list of atomic numbers corresponding to the provided atom symbols. 


---



### <kbd>function</kbd> `smiles_to_2d_graph`

```python
smiles_to_2d_graph(smiles)
```

Converts SMILES strings to 2D graph representations. 



**Args:**
 
 - <b>`smiles`</b> (str):  A SMILES string representing the molecule. 



**Returns:**
 
 - <b>`tuple[list[int], list[list[int]]]`</b>:  A tuple containing a list of atom IDs and a list of bonds represented as pairs of atom indices. 


## <kbd>module</kbd> `metrics`

These Python functions are designed to calculate various metrics for classification and regression tasks, particularly focusing on evaluating the performance of models on tasks involving predictions with associated uncertainties. 


---



### <kbd>function</kbd> `classification_metrics`

```python
classification_metrics(preds, lbs, masks, exclude: list = None)
```

Calculates various metrics for classification tasks. 

This function computes ROC-AUC, PRC-AUC, ECE, MCE, NLL, and Brier score for classification predictions. 



**Args:**
 
 - <b>`preds`</b> (numpy.ndarray):  Predicted probabilities for each class. 
 - <b>`lbs`</b> (numpy.ndarray):  Ground truth labels. 
 - <b>`masks`</b> (numpy.ndarray):  Masks indicating valid data points for evaluation. 
 - <b>`exclude`</b> (list, optional):  List of metrics to exclude from the calculation. 



**Returns:**
 
 - <b>`dict`</b>:  A dictionary containing calculated metrics. 


---



### <kbd>function</kbd> `regression_metrics`

```python
regression_metrics(preds, variances, lbs, masks, exclude: list = None)
```

Calculates various metrics for regression tasks. 

Computes RMSE, MAE, NLL, and calibration error for regression predictions and their associated uncertainties. 



**Args:**
 
 - <b>`preds`</b> (numpy.ndarray):  Predicted values (means). 
 - <b>`variances`</b> (numpy.ndarray):  Predicted variances. 
 - <b>`lbs`</b> (numpy.ndarray):  Ground truth values. 
 - <b>`masks`</b> (numpy.ndarray):  Masks indicating valid data points for evaluation. 
 - <b>`exclude`</b> (list, optional):  List of metrics to exclude from the calculation. 



**Returns:**
 
 - <b>`dict`</b>:  A dictionary containing calculated metrics. 


---



### <kbd>function</kbd> `regression_calibration_error`

```python
regression_calibration_error(lbs, preds, variances, n_bins=20)
```

Calculates the calibration error for regression tasks. 

Uses the predicted means and variances to estimate the calibration error across a specified number of bins. 



**Args:**
 
 - <b>`lbs`</b> (numpy.ndarray):  Ground truth values. 
 - <b>`preds`</b> (numpy.ndarray):  Predicted values (means). 
 - <b>`variances`</b> (numpy.ndarray):  Predicted variances. 
 - <b>`n_bins`</b> (int, optional):  Number of bins to use for calculating calibration error. Defaults to 20. 



**Returns:**
 
 - <b>`float`</b>:  The calculated calibration error. 

## <kbd>module</kbd> `io`

### <kbd>function</kbd> `set_log_path`

```python
set_log_path(args, time)
```

Sets up the log path based on given arguments and time. 



**Args:**
 
 - <b>`args`</b>:  Command-line arguments or any object with attributes `dataset_name`, `model_name`, `feature_type`, and `uncertainty_method`. 
 - <b>`time`</b> (str):  A string representing the current time or a unique identifier for the log file. 



**Returns:**
 
 - <b>`str`</b>:  The constructed log path. 


---



### <kbd>function</kbd> `set_logging`

```python
set_logging(log_path: Optional[str] = None)
```

Sets up logging format and file handler. 



**Args:**
 
 - <b>`log_path`</b> (Optional[str]):  Path where to save the logging file. If None, no log file is saved. 


---



### <kbd>function</kbd> `logging_args`

```python
logging_args(args)
```

Logs model arguments. 



**Args:**
 
 - <b>`args`</b>:  The arguments to be logged. Can be an argparse Namespace or similar object. 


---



### <kbd>function</kbd> `remove_dir`

```python
remove_dir(directory: str)
```

Removes a directory and its subtree. 



**Args:**
 
 - <b>`directory`</b> (str):  The directory to remove. 


---



### <kbd>function</kbd> `init_dir`

```python
init_dir(directory: str, clear_original_content: Optional[bool] = True)
```

Initializes a directory. Clears content if specified and directory exists. 



**Args:**
 
 - <b>`directory`</b> (str):  The directory to initialize. 
 - <b>`clear_original_content`</b> (Optional[bool]):  Whether to clear the original content of the directory if it exists. 


---



### <kbd>function</kbd> `save_json`

```python
save_json(obj, path: str, collapse_level: Optional[int] = None)
```

Saves an object to a JSON file. 



**Args:**
 
 - <b>`obj`</b>:  The object to save. 
 - <b>`path`</b> (str):  The path to the file where the object will be saved. 
 - <b>`collapse_level`</b> (Optional[int]):  Specifies how to prettify the JSON output. If set, collapses levels greater than this. 


---



### <kbd>function</kbd> `prettify_json`

```python
prettify_json(text, indent=2, collapse_level=4)
```

Prettifies JSON text by collapsing indent levels higher than `collapse_level`. 



**Args:**
 
 - <b>`text`</b> (str):  Input JSON text. 
 - <b>`indent`</b> (int):  The indentation value of the JSON text. 
 - <b>`collapse_level`</b> (int):  The level from which to stop adding new lines. 



**Returns:**
 
 - <b>`str`</b>:  The prettified JSON text. 


---



### <kbd>function</kbd> `convert_arguments_from_argparse`

```python
convert_arguments_from_argparse(args)
```

Converts argparse Namespace to transformers-style arguments. 



**Args:**
 
 - <b>`args`</b>:  argparse Namespace object. 



**Returns:**
 
 - <b>`str`</b>:  Transformers style arguments string. 


---



### <kbd>function</kbd> `save_results`

```python
save_results(path, preds, variances, lbs, masks)
```

Saves prediction results to a file. 



**Args:**
 
 - <b>`path`</b> (str):  Path where to save the results. 
 - <b>`preds`</b>:  Predictions to save. 
 - <b>`variances`</b>:  Variances associated with predictions. 
 - <b>`lbs`</b>:  Ground truth labels. 
 - <b>`masks`</b>:  Masks indicating valid entries. 


---



### <kbd>function</kbd> `load_results`

```python
load_results(result_paths: list[str])
```

Loads prediction results from files. 



**Args:**
 
 - <b>`result_paths`</b> (list[str]):  Paths to the result files. 



**Returns:**
 
 - <b>`tuple`</b>:  Predictions, variances, labels, and masks loaded from the files. 


---



### <kbd>function</kbd> `load_lmdb`

```python
load_lmdb(data_path, keys_to_load: list[str] = None, return_dict: bool = False)
```

Loads data from an LMDB file. 



**Args:**
 
 - <b>`data_path`</b> (str):  Path to the LMDB file. 
 - <b>`keys_to_load`</b> (list[str], optional):  Specific keys to load from the LMDB file. Loads all keys if None. 
 - <b>`return_dict`</b> (bool):  Whether to return a dictionary of loaded values. 



**Returns:**
 
 - <b>`dict or tuple`</b>:  Loaded values from the LMDB file. The format depends on `return_dict`. 


---



### <kbd>function</kbd> `load_unimol_preprocessed`

```python
load_unimol_preprocessed(data_dir: str)
```

Loads preprocessed UniMol dataset splits from an LMDB file. 



**Args:**
 
 - <b>`data_dir`</b> (str):  Directory containing the LMDB dataset splits. 



**Returns:**
 
 - <b>`dict`</b>:  Loaded dataset splits (train, valid, test). 

