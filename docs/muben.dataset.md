<!-- markdownlint-disable -->


# <kbd>module</kbd> `muben.dataset`
This module includes base classes for dataset creation and batch processing. 


---


## <kbd>function</kbd> `pack_instances`

```python
pack_instances(**kwargs) → list[dict]
```

Converts lists of attributes into a list of data instances. 

Each data instance is represented as a dictionary with attribute names as keys and the corresponding data point values as values. 



**Args:**
 
 - <b>`**kwargs`</b>:  Variable length keyword arguments, where each key is an attribute name and its value is a list of data points. 



**Returns:**
 
 - <b>`List[Dict]`</b>:  A list of dictionaries, each representing a data instance. 


---


## <kbd>function</kbd> `unpack_instances`

```python
unpack_instances(instance_list: list[dict], attr_names: list[str] = None)
```

Converts a list of dictionaries (data instances) back into lists of attribute values. 

This function is essentially the inverse of `pack_instances`. 



**Args:**
 
 - <b>`instance_list`</b> (List[Dict]):  A list of data instances, where each instance is a dictionary with attribute names as keys. 
 - <b>`attr_names`</b> ([List[str]], optional):  A list of attribute names to extract. If not provided, all attributes found in the first instance are used. 



**Returns:**
 
 - <b>`List[List]`</b>:  A list of lists, where each sublist contains all values for a particular attribute across all instances. 


---

## <kbd>class</kbd> `Batch`
Represents a batch of data instances, where each instance is initialized with attributes provided as keyword arguments. 

Each attribute name acts as a key to its corresponding value, allowing for flexible data handling within a batched context. 



**Attributes:**
 
 - <b>`size`</b> (int):  The size of the batch. Defaults to 0. 
 - <b>`_tensor_members`</b> (dict):  A dictionary to keep track of tensor attributes for device transfer operations. 


### <kbd>function</kbd> `__init__`

```python
__init__(**kwargs)
```

Initializes a Batch object with dynamic attributes based on the provided keyword arguments. 



**Args:**
 
 - <b>`**kwargs`</b>:  Arbitrary keyword arguments representing attributes of data instances within the batch.  A special keyword 'batch_size' can be used to explicitly set the batch size. 




---


### <kbd>function</kbd> `to`

```python
to(device)
```

Moves all tensor attributes to the specified device (cpu, cuda). 



**Args:**
 
 - <b>`device`</b>:  The target device to move the tensor attributes to. 



**Returns:**
 
 - <b>`self`</b>:  The batch instance with its tensor attributes moved to the specified device. 


---

## <kbd>class</kbd> `Dataset`
Custom Dataset class to handle data storage, manipulation, and preprocessing operations. 



**Attributes:**
 
 - <b>`_smiles`</b> (Union[list[str], None]):  Chemical structures represented as strings. 
 - <b>`_lbs`</b> (Union[np.ndarray, None]):  data labels. 
 - <b>`_masks`</b> (Union[np.ndarray, None]):  Data masks. 
 - <b>`_ori_ids`</b> (Union[np.ndarray, None]):  Original IDs of the datapoints,  specifically used for randomly split datasets. 
 - <b>`data_instances`</b>:  Packed instances of data. 


---

#### <kbd>property</kbd> data_instances

Returns the current data instances, considering whether the full dataset or a selection is being used. 

---

#### <kbd>property</kbd> lbs

Returns the label data, considering whether standardized labels are being used. 

---

#### <kbd>property</kbd> smiles

Returns the chemical structures represented as strings. 


---


### <kbd>function</kbd> `add_sample_by_ids`

```python
add_sample_by_ids(ids: list[int] = None)
```

Appends a subset of data instances to the selected data instances. 



**Args:**
 
 - <b>`ids`</b> (list[int], optional):  Indices of the selected instances. 



**Raises:**
 
 - <b>`ValueError`</b>:  If `ids` is not specified. 



**Returns:**
 
 - <b>`self`</b> (Dataset):  The dataset with the added data instances. 

---


### <kbd>function</kbd> `create_features`

```python
create_features(config)
```

Creates data features. This method should be implemented by subclasses to generate data features according to different descriptors or fingerprints. 



**Raises:**
 
 - <b>`NotImplementedError`</b>:  This method should be implemented by subclasses. 

---


### <kbd>function</kbd> `downsample_by`

```python
downsample_by(file_path: str = None, ids: list[int] = None)
```

Downsamples the dataset to a subset with the specified indices. 



**Args:**
 
 - <b>`file_path`</b> (str, optional):  Path to the file containing the indices of the selected instances. 
 - <b>`ids`</b> (list[int], optional):  Indices of the selected instances. 



**Raises:**
 
 - <b>`ValueError`</b>:  If neither `ids` nor `file_path` is specified. 



**Returns:**
 
 - <b>`self`</b> (Dataset):  The downsampled dataset. 

---


### <kbd>function</kbd> `get_instances`

```python
get_instances()
```

Gets the instances of the dataset. This method should be implemented by subclasses to pack data, labels, and masks into data instances. 



**Raises:**
 
 - <b>`NotImplementedError`</b>:  This method should be implemented by subclasses. 

---


### <kbd>function</kbd> `load`

```python
load(file_path: str)
```

Loads the entire dataset from disk. 



**Args:**
 
 - <b>`file_path`</b> (str):  Path to the saved file. 



**Returns:**
 self (Dataset) 

---


### <kbd>function</kbd> `prepare`

```python
prepare(config, partition, **kwargs)
```

Prepares the dataset for training and testing. 



**Args:**
 
 - <b>`config`</b>:  Configuration parameters. 
 - <b>`partition`</b> (str):  The dataset partition; should be one of 'train', 'valid', 'test'. 



**Raises:**
 
 - <b>`ValueError`</b>:  If `partition` is not one of 'train', 'valid', 'test'. 



**Returns:**
 
 - <b>`self`</b> (Dataset):  The prepared dataset. 

---


### <kbd>function</kbd> `read_csv`

```python
read_csv(data_dir: str, partition: str)
```

Reads data from CSV files. 



**Args:**
 
 - <b>`data_dir`</b> (str):  The directory where data files are stored. 
 - <b>`partition`</b> (str):  The dataset partition ('train', 'valid', 'test'). 



**Raises:**
 
 - <b>`FileNotFoundError`</b>:  If the specified file does not exist. 



**Returns:**
 self (Dataset) 

---


### <kbd>function</kbd> `save`

```python
save(file_path: str)
```

Saves the entire dataset for future use. 



**Args:**
 
 - <b>`file_path`</b> (str):  Path to the save file. 



**Returns:**
 self (Dataset) 

---


### <kbd>function</kbd> `set_standardized_lbs`

```python
set_standardized_lbs(lbs)
```

Sets standardized labels and updates the instance list accordingly. 



**Args:**
 
 - <b>`lbs`</b>:  The standardized label data. 



**Returns:**
 
 - <b>`self`</b> (Dataset):  The dataset with the standardized labels set. 

---


### <kbd>function</kbd> `toggle_standardized_lbs`

```python
toggle_standardized_lbs(use_standardized_lbs: bool = None)
```

Toggle between using standardized and unstandardized labels. 



**Args:**
 
 - <b>`use_standardized_lbs`</b> (bool, optional):  Whether to use standardized labels. Defaults to None. 



**Returns:**
 
 - <b>`self`</b> (Dataset):  The dataset with the standardized labels toggled. 

---


### <kbd>function</kbd> `update_lbs`

```python
update_lbs(lbs)
```

Updates the dataset labels and the instance list accordingly. 



**Args:**
 
 - <b>`lbs`</b>:  The new labels. 



**Returns:**
 
 - <b>`self`</b> (Dataset):  The dataset with the updated labels. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
