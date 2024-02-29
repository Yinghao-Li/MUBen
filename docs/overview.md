# Project Overview


The `muben` package consists of several sub-packages, each tailored for specific functionalities within the backbond and/or UQ method trainig workflow.
Below is a detailed overview of each sub-package and its purpose:

## `muben.args`
Manages command-line arguments and configuration settings for the `muben` package.

- Command-line argument/configuration file parsing
- Parameter validation and defaults

## `muben.dataset`
Provides functionality for loading, processing, and handling datasets.

- Dataset loading
- Feature generation
- Data preparation for `dataloader`
- Batching functions
- Collating functions

## `muben.layers`
Defines the output layer for compatibility with various objects (classification/regression), number of tasks, and UQ method (especially for Bayes-by-Backprop).

- Custom the output layer

## `muben.model`
Focuses on model definition and implementation.

- Model architecture and forward functions
- Loading functions for pre-trained model weights

## `muben.uncertainty`
Specializes in uncertainty estimation model architecture and training schemes.
Notice that some UQ methods without special training steps or the need for modifying backbone layers are directly incorporated in the trainer.

- Uncertainty estimation functions
- Uncertainty estimation training schemes


## `muben.train`
Dedicated to the training process, including batch processing, epoch management, and callback functionalities.
This module ensures efficient and effective model training, providing a robust framework for different training regimes.

- Training loops and batch processing
- Callbacks and training hooks
- Training metrics and evaluation

## `muben.utils`
Offers a collection of utility functions and helper tools that support the broader functionality of the `muben` package.
This module includes miscellaneous functionalities such as logging, data manipulation, and performance metrics.

- Logging and debugging tools
- Data manipulation utilities
- Performance and evaluation metrics
