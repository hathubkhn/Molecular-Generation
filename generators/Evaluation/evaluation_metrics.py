import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import entropy
from fcd import get_fcd
from tqdm import tqdm

class Evaluator:
    def __init__(self, train_smiles, gen_smiles):
        self.train_smiles = train_smiles
        self.gen_smiles = gen_smiles
    
    def compute_validity(self):
        valid_smiles = []
        for smiles in self.gen_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    valid_smiles.append(Chem.MolToSmiles(largest_mol))
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetMolFrags")
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
        validity_ratio = len(valid_smiles) / len(self.gen_smiles)
        return valid_smiles, validity_ratio
    
    def compute_uniqueness(self, valid_smiles):
        """ valid_smiles: list of SMILES strings."""
        unique_smiles = list(set(valid_smiles))
        uniqueness_ratio = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0
        return unique_smiles, uniqueness_ratio
    
    def compute_novelty(self, unique_smiles):
        """ unique_smiles: list of unique SMILES strings."""
        if self.train_smiles is None:
            print("Dataset SMILES is None, novelty computation skipped")
            return 1, 1
        
        novel = [smiles for smiles in unique_smiles if smiles not in self.train_smiles]
        novelty_ratio = len(novel) / len(unique_smiles) if unique_smiles else 0
        return novelty_ratio
    
    def compute_fcd(self):
        return get_fcd(self.train_smiles, self.gen_smiles)
    
    def compute_kl_div(self, n_bits=2048):
        def fingerprint_distribution(smiles):
              fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(sm), 2, nBits=n_bits)
                    for sm in smiles if Chem.MolFromSmiles(sm)]
              if len(fps) == 0:
                    return np.zeros(n_bits)
              counts = np.sum(fps, axis=0)
              return counts / np.sum(counts)
    
        p_train = fingerprint_distribution(self.train_smiles)
        q_gen = fingerprint_distribution(self.gen_smiles)
    
        return entropy(p_train, q_gen) if np.any(q_gen) else float('inf')
    
    def cal_metrics(self):
        print("Calculating validity...")
        valid_smiles, validity = self.compute_validity()
        print(f"Calculating uniqueness over {len(valid_smiles)} valid SMILES...")
        unique_smiles, uniqueness = self.compute_uniqueness(valid_smiles)
        print(f"Calculating novelty over {len(unique_smiles)} unique SMILES...")
        novelty = self.compute_novelty(unique_smiles)
        print("Calculating FCD...")
        fcd_ = self.compute_fcd()
        print("Calculating KL-Div...")
        kl_div = self.compute_kl_div()
        
        return {
              "Validity": validity,
              "Uniqueness": uniqueness,
              "Novelty": novelty,
              "FCD": fcd_,
              "KL-Divergence": kl_div
        }


if __name__ == "__main__":
    train_smiles_fpath = "smiles_files/zinc_train_800k.txt"
    gen_smiles_fpath = "smiles_files/smiles_digress_5k.txt"
    
    with open(train_smiles_fpath, 'r') as train_f:
        train_lines = train_f.readlines()
    
    with open(gen_smiles_fpath, 'r') as gen_f:
        gen_lines = gen_f.readlines()
    
    train_smiles = [line.strip() for line in train_lines]
    gen_smiles = [line.strip() for line in gen_lines]
    
    print(f"Calculating metrics on {len(gen_smiles)} generated molecules and {len(train_smiles)} reference molecules")
    
    evaluator = Evaluator(train_smiles=train_smiles, gen_smiles=gen_smiles)
    
    res = evaluator.cal_metrics()

    print(">>>> Results:", res)