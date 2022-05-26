# Deep Surrogate Docking: Accelerating Automated Drug Discovery with Graph Neural Networks

This repository is the official implementation of Deep Surrogate Docking: Accelerating Automated Drug Discovery with Graph Neural Networks. 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Note: Pytorch geometric may require a seperate installation process, depending on the system being used. If Pytorch Geometric cannot be installed using the requirements file, refer to the [Pytorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install Pytorch Geometric, and then use the requirements file to install all other dependencies.

## Data
TODO

## Training

To train the model(s) in the paper, modify the configuration file `./graph-dock/config.json` and run this command:

```train
python train.py
```

The provided `./graph-dock/config.json` file contains the default hyperparameters used to obtain the FiLMv2 results in our work. 

## Evaluation

To evaluate a model on a subset of the ZINC dataset, run:

```eval
python inference.py 
```

Note that this script also depends on the `./graph-dock/config.json` file, which currently contains the default parameters used to obtain the results in our work. 

## Pre-trained Models

Links to pretrained models are provided in the results table. Models should be downloaded and put in `./checkpoints/MODEL_NAME`. An example checkpoint is provided in this repository.

# - [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 


## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name | W-MSE | RES Score |
| ---------- | ----- | --------- |
| GIN        | TODO  | TODO      |
| GAT        | TODO  | TODO      |
| FiLM       | TODO  | TODO      |
| FiLMv2     | TODO  | TODO      |


## Contributing


