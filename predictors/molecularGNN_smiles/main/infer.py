import torch
import numpy as np
import sys
import preprocess as pp
from train import MolecularGraphNeuralNetwork
from rdkit import Chem

def load_model(model_path, N_fingerprints, dim, layer_hidden, layer_output, device):
    model = MolecularGraphNeuralNetwork(N_fingerprints, dim, layer_hidden, layer_output).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device).state_dict())
    model.eval()
    return model

def predict_smiles(smiles, model, radius, device):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES input.")

    data = pp.create_single_molecule_data(mol, radius, device)
    data_batch = list(zip(*[data]))

    predicted = model.forward_regressor_infer(data_batch, train=False)
    return float(predicted)

if __name__ == "__main__":
    task = 'regression'
    smiles = 'CC[C@@H]1CN(C(=O)c2cc(Cn3c(=O)[nH]c(=O)c4ccccc43)ccc2F)CCN1CC(F)(F)F'   
    dataset = 'pic50_augment_20k'
    radius = 1   
    dim = 50   
    layer_hidden = 6
    layer_output = 6
    model_path = './../ckpt/pic50.pth' 
    N_fingerprints = 324

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path, N_fingerprints, dim, layer_hidden, layer_output, device)

    predicted_value = predict_smiles(smiles, model, radius, device)

    print(f"Predicted value for SMILES '{smiles}': {predicted_value:.4f}")
