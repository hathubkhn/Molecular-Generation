import os
import os.path as osp
import sys
import yaml
import argparse
from rdkit import Chem
from easydict import EasyDict as edict
import torch

class GNNPredictor():
    def __init__(self, cfg_dir):
        self.cfg = edict(yaml.load(open(osp.join(cfg_dir, "gnn.yaml"), "r"), Loader=yaml.FullLoader))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from train import MolecularGraphNeuralNetwork
        model = MolecularGraphNeuralNetwork(self.cfg.N_fingerprints, 
                                                 self.cfg.dim, 
                                                 self.cfg.layer_hidden, 
                                                 self.cfg.layer_output).to(self.device)
        state_dict = torch.load(self.cfg.model_path, map_location=self.device)#.state_dict()
        model.load_state_dict(state_dict)
        self.model = model
        self.model.eval()

    def predict(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES input.")

        import preprocess
        data = preprocess.create_single_molecule_data(mol, self.cfg.radius, self.device)
        data_batch = list(zip(*[data]))
    
        predicted = self.model.infer_regressor(data_batch, train=False)
        return float(predicted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or generate SMILES with Generators")
    parser.add_argument("--model", type=str, required=True, choices=["gnn", "lstm", "mlp", "ml"], help="Choose pic50 prediction model")
    args = parser.parse_args()

    if args.model == "gnn":
        sys.path.append("./predictors/molecularGNN_smiles/main/")
        predictor = GNNPredictor(cfg_dir="./configs/gnn")
	
    smiles = "Cn1c2cccc(C(N)=O)c2c(=O)n1C1CCN(C2CCC(F)(F)CC2)CC1"
    print(predictor.predict(smiles))