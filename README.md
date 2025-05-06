# Diploma Thesis

This repository contains the source code for my diploma thesis.
This thesis evaluates compositionality in language learning within a simulated robotic environment. 
Specifically, we created a model that combines a visual model (ResNet-18) with an echo state network (ESN) to process sequences of frames capturing a robotic arm interacting with objects on a table. 
These sequences, generated in a robotic simulator, provide both visual and proprioceptive information, such as joint positions. 
The goal is to train the model to recognise and describe actions, objects, and their colours (e.g., “push left green apple”). 

- `DP_model1/`: Contains Python scripts for training and evaluating the model where the visual model (**ResNet-18**) is **not fine-tuned** during training. The ESN is implemented using the [ReservoirPy](https://reservoirpy.readthedocs.io/en/latest/) library.

- `DP_model2/`: Contains scripts for a second model where the visual model **is fine-tuned**. The ESN is implemented using the [PyTorch-ESN](https://github.com/stefanonardo/pytorch-esn.git) module.

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Running the training script, the required argument is the path to the dataset::
```bash
python3 DP_model1/train.py --data_path path/to/the/dataset
python3 DP_model2/train.py --data_path path/to/the/dataset
```

To view additional commandline arguments:

```bash
python3 DP_model1/train.py -h
python3 DP_mode2/train.py -h
```
