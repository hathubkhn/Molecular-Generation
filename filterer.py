from tdc import Oracle
from rdkit import Chem
from rdkit.Chem import Draw
import sys
from tqdm import tqdm
import os
import os.path as osp
from predictor import GNNPredictor

class SMILESFilterer():
    def __init__(self):
        self.logP_predictor = Oracle(name = 'logP')
        self.SA_predictor = Oracle(name = 'SA')

        sys.path.append("./predictors/molecularGNN_smiles/main/")
        self.pic50_predictor = GNNPredictor(cfg_dir="./configs/gnn")

    def count_large_rings(self, smiles, max_atoms_per_ring=6):
       try:
           mol = Chem.MolFromSmiles(smiles)
           if mol is None:
               return None
           ring_info = mol.GetRingInfo()
           ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
           large_ring_count = sum(1 for size in ring_sizes if size >= max_atoms_per_ring)
           return large_ring_count
       except:
           return None

    def cal_properties(self, smiles):
        SA = self.SA_predictor(smiles)
        logP = self.logP_predictor(smiles)
        pic50 = self.pic50_predictor.predict(smiles)
        large_ring_count = self.count_large_rings(smiles)
    
        return logP, SA, pic50, large_ring_count

    def filter_lst_smiles(self, list_smiles, lower_logP=1, upper_logP=4, lower_SA=1, upper_SA=3, lower_pic50=8, upper_pic50=20, max_rings_count=1, save_dir="./filtered_smiles",  save_img_smiles=False):  
        filtered_smiles_count = 0
        filtered_smiles = []
        count = 0
        for smiles in tqdm(list_smiles):     
            logP, SA, pic50, large_ring_count = self.cal_properties(smiles)
            if lower_logP <= logP <= upper_logP and lower_SA <= SA <= upper_SA and lower_pic50 <= pic50 <= upper_pic50 and large_ring_count <= max_rings_count:
                filtered_smiles_count += 1
                filtered_smiles.append((smiles, logP, SA, pic50))
    
        print("Num filtered smiles: ", filtered_smiles_count)
    
        top_sas = sorted(filtered_smiles, key=lambda x: x[2])

        if save_img_smiles:
            os.makedirs(save_dir, exist_ok=True)
            for smiles, logP, SA in top_sas:
                save_name = f"{smiles}_{round(logP, 3)}_{round(SA, 3)}_{round(pic50, 3)}.jpg"
                save_path = osp.join(save_dir, save_name)
                mol = Chem.MolFromSmiles(smiles)
                img = Draw.MolToImage(mol)
                img.save(save_path)
    
            print(f">> Saved {filtered_smiles_count} molecules to {save_dir}")

        return top_sas # [(smiles, logP, SA, pic50)]


if __name__ == "__main__":
    filterer = SMILESFilterer()

    smiles_fpath = "smiles_mood_1k.txt"
    with open(smiles_fpath, "r") as smiles_f:
        lines = smiles_f.readlines()

    lst_smiles = [line.strip() for line in lines]

    print(filterer.filter_lst_smiles(lst_smiles))