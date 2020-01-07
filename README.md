# Extended Network: Combining deep learning and statistical language models
This repository holds code for Extended Network, Pointer-Mixture network and PHOG.

## Introduction

Extended Network has been developed to combine deep learning and probabilistic language models for next node value predictions in abstract syntax trees (ASTs).
This implementation features an Extended Network consisting of a Pointer-Mixture network and a PHOG with the goal 
to reduce errors with OoV words with long-range dependencies.

## Data set
To train and evalute this Extended Network we used the 150k Python data-set from Eth Zürich available at: https://www.sri.inf.ethz.ch/py150. The data-set consists of Python ASTs collected from GitHub and is split into 100k ASTs for training and 100k ASTs for testing. Download the data-set with:
```python
python3 setup.py
```

Then run the following command at the folder "neural_code_completion\preprocess_code":
```python
python3 build_dataset.py
```

## Description of the directories

#### neural_code_completion/models
Holds the different models and dataloaders:
- config.py: holds the configurations for Pointer-Mixture and Extended Network
- extendedNetwork.py: class defining the Extended Network
- pointerMixture.py: class defining the Pointer-Mixture network
- reader_pointer_extended.py: loads the terminal and non-terminal corpora for the Extended Network
- reader_pointer_original.py: loads the terminal and non-terminal corpora for the Pointer-Mixture network
- train_extended.py: trains the Extended Network
- train_pointer_mixture.py: trains the Pointer-Mixture network

#### neural_code_completion/preprocess_code
Data pre-processing consists of turning ASTs into IDs and saving them along with metadata into pickle files
- build_dataset.py: creates the train development data-set (random seed 42 to reproduce our data-set)
- freq_dict.py: creates the frequency dictionary used for creating the terminal vocabulary
- get_non_terminal.py: creates the non-terminals corpus
- get_terminal_extended.py: creates the terminal corpus for the Extended Network
- get_terminal_original.py: creates the terminal corpus for the Pointer-Mixture network
- get_total_length.py, utils.py: utilities and helper functions

## Training Extended Network
To train Extended Network first create the pickle files with the pre-processing code located in neural_code_completion/preprocess_code. Alternatively download the pre-processed pickle files here: [pickle_data](https://drive.google.com/open?id=1PJ-rOMOOT7KzaM203Shs-X2EE-oFv-r0). The pickle files should be put under the folder "neural_code_completion\pickle_data"

Thereafter set custom flags in train_extended.py or change the used configuration in config.py.
Then simply run:
```python
python3 train_extended.py
```
