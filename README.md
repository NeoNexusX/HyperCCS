# HyperCCS: Large MultiModal Model

# Model for CCS 

<img src="https://s2.loli.net/2025/05/30/4RXBx7LJq3zeyi9.png" alt="TOC2-01.png" style="zoom: 50%;" />

# Prediction

This repository contains the implementation of a deep learning model for predicting Collision Cross Section (CCS) values of molecules. The model utilizes a transformer-based architecture with molecular structure information for accurate CCS predictions.

## Getting Started

### Clone the Repository

```bash
git lfs install  # make sure install git lfs
git clone https://github.com/NeoNexusX/HyperCCS.git
cd HyperCCS
git lfs pull    # download ckpt files
```

## Project Structure

```bash
.
├── data/                   # Processed data directory
├── data_prepare/          # Data preparation scripts and utilities
├── model/                 # Model architecture implementation
├── Pretrained MoLFormer/  # Pre-trained model checkpoints and configurations
├── original_data/         # Original dataset files
├── predict.py            # Script for making predictions
├── main.py              # Main training script
├── finetune.sh          # Shell script for model fine-tuning
└── data_prepare_example.py # Example of data preparation
```

## Environment Setup

1. Create a new Python environment:
```bash
conda create -n hyperccs python=3.8.10
conda activate hyperccs
```

2. Install the required packages:
```bash
conda install pytorch==1.7.1 cudatoolkit=11.0 -c pytorch
conda install numpy=1.22.3 pandas=1.2.4 scikit-learn=0.24.2 scipy=1.6.2
conda install rdkit==2022.03.2 -c conda-forge
```

```bash
pip install transformers==4.6.0 pytorch-lightning==1.1.5 pytorch-fast-transformers==0.4.0 datasets==1.6.2 jupyterlab==3.4.0 ipywidgets==7.7.0 bertviz==1.4.0
```

```bash
git clone https://github.com/NVIDIA/apex
cd apex
git checkout tags/22.03 -b v22.03
export CUDA_HOME='Cuda 11 install path'
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

## Model Training and Fine-tuning

### Fine-tuning Script Usage

To fine-tune the model on your own dataset, use the `finetune.sh` script. You can customize various parameters to optimize the training process:

```bash
./finetune.sh
```

#### Available Parameters

Model Architecture Parameters:
- `--n_head`: Number of attention heads (default: 12)
- `--n_layer`: Number of transformer layers (default: 12)
- `--n_embd`: Embedding dimension (default: 768)
- `--adduct_num`: Number of adduct types (default: 3)
- `--ecfp_num`: ECFP fingerprint size (default: 1024)
- `--type`: Fusion type ('early' or 'later') for feature integration

Training Parameters:
- `--device`: Computing device ('cuda' or 'cpu')
- `--batch_size`: Training batch size (default: 64)
- `--d_dropout`: Dropout rate for dense layers (default: 0.1)
- `--dropout`: General dropout rate (default: 0.1)
- `--lr_start`: Initial learning rate (default: 1e-5)
- `--max_epochs`: Maximum training epochs (default: 100)
- `--num_workers`: Number of data loading workers (default: 20)

Data and Checkpoint Parameters:
- `--dataset_name`: Name of the dataset (e.g., 'FD_M0')
- `--data_root`: Path to the data directory (e.g., './data/FD_M0')
- `--measure_name`: Target measurement name (default: 'CCS')
- `--seed_path`: Path to the pre-trained model checkpoint
- `--checkpoints_folder`: Directory to save model checkpoints
- `--project_name`: Name for the training run  **must have 'Attention' in the string to use the fusion module**
- `--checkpoint_every`: Save checkpoint every N epochs (default: 10)

## Making Predictions

To make predictions using the pre-trained model:

```bash
python predict.py --dataset_name METLIN
```

The script will:
1. Load the specified pre-trained model
2. Process the input data
3. Generate CCS predictions
4. Save results to `predictions_[dataset_name].csv` with the following columns:
   - true_ccs: Actual CCS values
   - predicted_ccs: Model predictions
   - smiles: SMILES representation of molecules
   - adducts: Adduct types

## Model Architecture

The model uses a transformer-based architecture with:
- SMILES tokenization for molecular representation
- Attention mechanisms for capturing molecular structure
- Support for different adduct types ([M+H]+, [M+Na]+, [M-H]-)
- Early and late fusion options for feature integration

## File Descriptions

- `predict.py`: Main prediction script
- `data_prepare.py`: Data preprocessing and splitting
- `model/layers/main_layer.py`: Core model implementation
- `bert_vocab.txt`: Vocabulary file for molecular tokenization
- `Pretrained MoLFormer/`: Contains model checkpoints and hyperparameters

## Citation

If you use this code in your research, please cite our work:
[Citation information to be added]

## License

This project is licensed under the terms of the LICENSE file included in the repository.

## Contact

For questions and issues, please open an issue in the GitHub repository.
