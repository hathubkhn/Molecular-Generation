import pandas as pd
from rdkit import Chem

def generate_random_smiles(smiles, num_augments=5):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    
    smiles_list = []
    for _ in range(num_augments):
        try:
            new_smiles = Chem.MolToSmiles(mol, doRandom=True)
        except:
            new_smiles = Chem.MolToSmiles(mol, canonical=True)
            
        if new_smiles != smiles:
            smiles_list.append(new_smiles)
    
    return smiles_list

if __name__ == "__main__":

    file_path = "./predictor/ic50_org.csv"
    output_path = "./predictor/ic50_augmented_40k.csv"
    NUM_AUGMENT = 20
    
    df = pd.read_csv(file_path)
    augmented_data = []
    for _, row in df.iterrows():
        original_smiles = row["Column1"]
        pic50 = row["Column2"]
        new_smiles_list = generate_random_smiles(original_smiles, num_augments=NUM_AUGMENT)
        
        for new_smiles in new_smiles_list:
            augmented_data.append([new_smiles, pic50])
    
    df_aug = pd.DataFrame(augmented_data, columns=["Column1", "Column2"])
    df_final = pd.concat([df, df_aug], ignore_index=True)
    df_final.to_csv(output_path, index=False)