# infer.py
import os
import torch
import torch.nn.functional as F
import numpy as np

from openchem.data.utils import smiles_to_tensor
from openchem.models.Smiles2Label import Smiles2Label
from openchem.modules.encoders.rnn_encoder import RNNEncoder
from openchem.modules.mlp.openchem_mlp import OpenChemMLP
from openchem.modules.embeddings.basic_embedding import Embedding
from openchem.utils.utils import identity
from openchem.data.smiles_data_layer import SmilesDataset
from openchem.data.utils import get_tokens

CHECKPOINT_PATH = './checkpoints/logP/fold_0/checkpoint/best_model.pth'
TOKENS = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']
TOKENS = ''.join(TOKENS) + ' '

N_HIDDEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_params = {
    'use_cuda': torch.cuda.is_available(),
    'task': 'regression',
    'embedding': Embedding,
    'embedding_params': {
        'num_embeddings': len(TOKENS),
        'embedding_dim': N_HIDDEN,
        'padding_idx': TOKENS.index(' ')
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

# checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

model.load_model(CHECKPOINT_PATH)


def predict_smiles(smiles: str):
    task = model.task
    use_cuda = model.use_cuda
    
    sample_batched = ...
    
    batch_input, batch_object = model.cast_inputs(sample_batched,
                                                          task,
                                                          use_cuda,
                                                          for_predction=True)
    with torch.no_grad():
        predicted = model(batch_input, eval=True)
    if hasattr(predicted, 'detach'):
        predicted = predicted.detach().cpu().numpy()
    return predicted


if __name__ == '__main__':

    smiles = "O=C1NC(=O)c2c1c1c(c3oc4ccccc4c23)CCC1" # 10.698 
    pred = predict_smiles(smiles)
    print(f"Predicted pIC50 for '{smiles}': {pred:.4f}")
