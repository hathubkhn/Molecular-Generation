import os
import sys
import copy
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

from openchem.models.Smiles2Label import Smiles2Label
from openchem.modules.encoders.rnn_encoder import RNNEncoder
from openchem.modules.mlp.openchem_mlp import OpenChemMLP
from openchem.modules.embeddings.basic_embedding import Embedding
from openchem.data.smiles_data_layer import SmilesDataset
from openchem.data.utils import (
    read_smiles_property_file, save_smiles_property_file, create_loader, get_tokens
)
from openchem.utils.utils import identity
from openchem.models.openchem_model import build_training, fit, evaluate

# Paths
DATA_PATH = '../data/ic50_augmented_50k.csv'
TMP_DATA_DIR = '../data/tmp/'
LOG_DIR = '../checkpoints/logP/'

# Model Hyperparameters
N_HIDDEN = 128
BATCH_SIZE = 128
NUM_EPOCHS = 100
LR = 0.005

# Prepare data
smiles_data = read_smiles_property_file(DATA_PATH, cols_to_read=[0, 1], keep_header=False)
pic50 = [float(item.replace(',', '')) for item in smiles_data[1]]
smiles = np.array(smiles_data[0])
labels = np.array(pic50)

# Define tokens
_, _, _ = get_tokens(smiles)
tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']
tokens = ''.join(tokens) + ' '

# Model parameters
model_params = {
    'use_cuda': True,
    'random_seed': 42,
    'world_size': 1,
    'task': 'regression',
    'data_layer': SmilesDataset,
    'use_clip_grad': False,
    'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS,
    'logdir': LOG_DIR,
    'print_every': 1,
    'save_every': 5,
    'train_data_layer': None,
    'val_data_layer': None,
    'eval_metrics': r2_score,
    'criterion': nn.MSELoss(),
    'optimizer': Adam,
    'optimizer_params': {'lr': LR},
    'lr_scheduler': ExponentialLR,
    'lr_scheduler_params': {'gamma': 0.98},
    'embedding': Embedding,
    'embedding_params': {
        'num_embeddings': len(tokens),
        'embedding_dim': N_HIDDEN,
        'padding_idx': tokens.index(' ')
    },
    'encoder': RNNEncoder,
    'encoder_params': {
        'input_size': N_HIDDEN,
        'layer': "LSTM",
        'encoder_dim': N_HIDDEN,
        'n_layers': 2,
        'dropout': 0.8,
        'is_bidirectional': False
    },
    'mlp': OpenChemMLP,
    'mlp_params': {
        'input_size': N_HIDDEN,
        'n_layers': 2,
        'hidden_size': [N_HIDDEN, 1],
        'activation': [F.relu, identity],
        'dropout': 0.0
    }
}

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TMP_DATA_DIR, exist_ok=True)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_data = list(kf.split(smiles, labels))

models, results = [], []

for i, (train_idx, test_idx) in enumerate(cv_data):
    print(f'Cross validation, fold number {i} in progress...')
    X_train, y_train = smiles[train_idx], labels[train_idx].reshape(-1)
    X_test, y_test = smiles[test_idx], labels[test_idx].reshape(-1)

    train_file = os.path.join(TMP_DATA_DIR, f'{i}_train.smi')
    test_file = os.path.join(TMP_DATA_DIR, f'{i}_test.smi')

    save_smiles_property_file(train_file, X_train, y_train.reshape(-1, 1))
    save_smiles_property_file(test_file, X_test, y_test.reshape(-1, 1))

    train_dataset = SmilesDataset(train_file, delimiter=',', cols_to_read=[0, 1], tokens=tokens, flip=False)
    test_dataset = SmilesDataset(test_file, delimiter=',', cols_to_read=[0, 1], tokens=tokens, flip=False)

    model_params['train_data_layer'] = train_dataset
    model_params['val_data_layer'] = test_dataset
    model_params['logdir'] = os.path.join(LOG_DIR, f'fold_{i}')

    os.makedirs(model_params['logdir'], exist_ok=True)
    os.makedirs(os.path.join(model_params['logdir'], 'checkpoint'), exist_ok=True)

    train_loader = create_loader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = create_loader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

    model = Smiles2Label(params=model_params).cuda()
    models.append(model)
    criterion, optimizer, lr_scheduler = build_training(model, model_params)
    result = fit(model, lr_scheduler, train_loader, optimizer, criterion, model_params, eval=True, val_loader=val_loader)
    results.append(result)

# Evaluation
rmse, r2_scores = [], []

for i in range(5):
    test_dataset = SmilesDataset(os.path.join(TMP_DATA_DIR, f'{i}_test.smi'),
                                 delimiter=',', cols_to_read=[0, 1], tokens=tokens, flip=False)
    val_loader = create_loader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)
    metrics = evaluate(models[i], val_loader, model_params['criterion'])
    rmse.append(np.sqrt(metrics[0]))
    r2_scores.append(metrics[1])

print("n_epochs:", NUM_EPOCHS)
print("Cross-validated RMSE:", np.mean(rmse))
print("Cross-validated R2 score:", np.mean(r2_scores))
