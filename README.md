# HeNCler

## Code Usage

### Download

First, navigate to the unzipped directory and prepare your environment. This is explained in the following section.

### Install packages in conda environment

Run the following in terminal. This will create a conda environment named *hencler*.

```
conda create --name hencler python=3.10 -y
```

Activate the conda environment with the command `conda activate hencler`. 
Install the torch, torch_geometric and pyg library according to your setup, following their installation instructions on
https://pytorch.org/get-started/locally/ and https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html (we used torch 2.0.0 and CUDA 11.7 for our experiments).

To install the remaining dependencies, run:
```R
pip install wcmatch nested_dict 
pip3 install -U scikit-learn
pip install hydra-core --upgrade
```

### Directories
Change your data and output directory in the definitions.py file if desired.

### Train

Activate the conda environment `conda activate hencler`.

The configuration is done using YAML files with [hydra](https://hydra.cc/). The default configuration is in the `conf` directory. The provided default_config.yaml gives the configuration of the reported experiments. 

To reproduce the reported experiments, simply choose a dataset (e.g. texas) and run the following command:
```R
python run_experiment.py d_name=texas
```
To change hyperparameters, you can create and add your own YAML files, or just add arguments to your command:
```R
python run_experiment.py --config-name my_own_yaml_file
python run_experiment.py d_name=chameleon num_cl=16 runs=1
```

