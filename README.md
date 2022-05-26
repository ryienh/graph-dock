# Deep Surrogate Docking: Accelerating Automated Drug Discovery with Graph Neural Networks

This repository is the official implementation of Deep Surrogate Docking: Accelerating Automated Drug Discovery with Graph Neural Networks. 

Please note that this is an anonymized version of the codebase for the purposes of review. The full reposity will be made publically available after the review period. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Note: Pytorch geometric may require a seperate installation process, depending on the system being used. If Pytorch Geometric cannot be installed using the requirements file, refer to the [Pytorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install Pytorch Geometric, and then use the requirements file to install all other dependencies.

## Data
The ZINC subset used in this project, along with the docking scores obtained by Lyu et al (2019), can be downloaded directly from the authors [here](https://figshare.com/articles/dataset/D4_screen_table_csv_gz/7359401). The default settings in `graph-dock/config.json` expect this data to have the following path `./data/d4_table_name_smi_energy_hac_lte_25_title.csv`. However, this can easily be modified in the configuration file. 

Before running any training or inference, this data needs to be preprocessed using the `preprocess_data` function in `util.py`. Invoking `util.py` as a script in the root directory of the repository will create a preprocessed version of the data consistent with current configuration settings in `graph-dock/config.json`. All other (in memory) preprocessing is handled automatically by the training script. 

## Training

To train the model(s) in the paper, modify the configuration file `./graph-dock/config.json` as needed and run this command:

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


## Results

Our model achieves the following performance on a witheld test partition of the ZINC dataset:

| Model name | W-MSE | RES Score |
| ---------- | ----- | --------- |
| GIN        | 0.402 | 0.742     |
| GAT        | 0.396 | 0.763     |
| FiLM       | 0.389 | 0.768     |
| FiLMv2     | 0.383 | 0.773     |

Please refer to our paper for more details.


## Contributing
We greatly welcome suggestions and contributions to our code! Please feel free to fork this repository, hack away, and submit a pull request.

