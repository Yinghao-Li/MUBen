<!-- markdownlint-disable -->

# <kbd>module</kbd> `muben.layers`

Versatile Output Layer for Backbone Models 

This module implements an output layer that can be applied to various backbone models. It provides the option of utilizing different uncertainty  methods for model's output and supports both classification and regression tasks. 


---

## <kbd>class</kbd> `OutputLayer`
Customizable output layer for various backbone models. 

This class provides an interface to add an output layer with or without uncertainty methods to various backbone models. It supports both classification and regression tasks and allows for the introduction of uncertainty into the model's output. 



**Args:**
 
 - <b>`last_hidden_dim`</b> (int):  Dimensionality of the last hidden state from the backbone model. 
 - <b>`n_output_heads`</b> (int):  Number of output heads (e.g., number of classes for classification). 
 - <b>`uncertainty_method`</b> (str, optional):  Method to introduce uncertainty in the output layer.  Available methods are defined in UncertaintyMethods. Defaults to UncertaintyMethods.none. 
 - <b>`task_type`</b> (str, optional):  Type of task - "classification" or "regression". Defaults to "classification". 
 - <b>`**kwargs`</b>:  Additional keyword arguments to be passed to the specific output layers. 



**Attributes:**
 
 - <b>`_uncertainty_method`</b> (str):  The uncertainty method used in the output layer. 
 - <b>`_task_type`</b> (str):  The type of task the model is configured for (classification or regression). 
 - <b>`output_layer`</b> (nn.Module):  The specific output layer instance used in the model. 
 - <b>`kld`</b> (Optional[torch.Tensor]):  Kullback-Leibler Divergence for Bayesian methods, if applicable. 


### <kbd>function</kbd> `__init__`

```python
__init__(
    last_hidden_dim,
    n_output_heads,
    uncertainty_method='none',
    task_type='classification',
    **kwargs
)
```

Initializes the OutputLayer instance. 



**Args:**
 
 - <b>`last_hidden_dim`</b> (int):  Dimensionality of the last hidden state from the backbone model. 
 - <b>`n_output_heads`</b> (int):  Number of output heads (e.g., number of classes for classification). 
 - <b>`uncertainty_method`</b> (str, optional):  Method to introduce uncertainty in the output layer.  Available methods are defined in UncertaintyMethods. Defaults to UncertaintyMethods.none. 
 - <b>`task_type`</b> (str, optional):  Type of task - "classification" or "regression". Defaults to "classification". 
 - <b>`**kwargs`</b>:  Additional keyword arguments to be passed to the specific output layers. 




---


### <kbd>function</kbd> `forward`

```python
forward(x: Tensor, **kwargs) → Tensor
```

Forward pass of the output layer. 



**Args:**
 
 - <b>`x`</b> (torch.Tensor):  Input tensor for the output layer. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  The output logits or values of the model. 

---


### <kbd>function</kbd> `initialize`

```python
initialize() → OutputLayer
```

Initializes the weights of the output layer. 

Different initializations are applied based on the uncertainty method and task type. 



**Returns:**
 
 - <b>`OutputLayer`</b>:  The initialized model instance. 




