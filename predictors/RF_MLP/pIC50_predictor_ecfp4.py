import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import os
import os.path as osp
import joblib

class MoleculeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.model(x).squeeze()

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns={"Column1": "Smiles", "Column2": "pIC50"})
    df = extract_features(df)
    
    X = np.stack(df["ECFP4"].values)
    X = np.hstack([X, df.drop(columns=["ECFP4", "pIC50"]).values])
    y = df["pIC50"].values
    
    return train_test_split(X, y, test_size=0.9, random_state=42)

def compute_ecfp4(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    else:
        return np.zeros(n_bits)

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return [
            Descriptors.MolLogP(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
        ]
    else:
        return [0, 0, 0, 0, 0]

def extract_features(df):
    df["ECFP4"] = df["Smiles"].apply(compute_ecfp4)
    descriptors = df["Smiles"].apply(compute_descriptors).apply(pd.Series)
    descriptors.columns = ["MolLogP", "NumHAcceptors", "NumHDonors", "NumRotatableBonds", "RingCount"]
    df = pd.concat([df, descriptors], axis=1)
    df.drop(columns=["Smiles"], inplace=True)
    return df


def train_ml_models(X_train, y_train, X_test, y_test, save_path="saved_models"):

    param_grids = {
        "RandomForest": {
            "n_estimators": [100, 300, 500],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [1.0, "sqrt"]
        },
        "XGBoost": {
            "n_estimators": [100, 300, 500],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 6, 10],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "reg_alpha": [0, 0.01, 0.1],
            "reg_lambda": [0, 0.01, 0.1]
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "epsilon": [0.01, 0.1, 0.5],
            "kernel": ["linear", "rbf"]
        }
    }
    
    models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "SVM": SVR()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        pcc, _ = pearsonr(y_test, y_pred)
    
        results[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "PCC": pcc
        }
    
        print(results[name])
        
        model_path = osp.join(save_path, f"{name}_best_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"Saved best model for {name} at {model_path}")
        
def train_deep_learning(X_train, y_train, X_test, y_test, save_path="saved_models"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data = MoleculeDataset(X_train, y_train)
    test_data = MoleculeDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    epochs = 500
    lr = 0.001

    model = MLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_mse = float("inf")
    model_path = ""

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
    
        batch_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
    
        for X_batch, y_batch in batch_loader:
            X_batch, y_batch = X_batch.float().to(device), y_batch.float().to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
            batch_loader.set_postfix(loss=loss.item())
    
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}")
    
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                y_pred_test = model(X_test_tensor).squeeze().cpu().numpy()
    
            mse = mean_squared_error(y_test, y_pred_test)
            print(f"...Validation: Epoch {epoch+1} - MSE: {mse:.4f}")
    
            if mse < best_mse:
                best_mse = mse
                model_path = osp.join(save_path, "best_deep_weight.pth")
                torch.save(model.state_dict(), model_path)
                print(f"✅ Saved best model at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model(X_test_tensor).squeeze().cpu().numpy()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pcc, _ = pearsonr(y_test, y_pred)

    print({
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "PCC": pcc,
        "Best Weights": model_path
    })

def test_deeplearning(X_test, y_test, model_path):
    print(f"Testing model from {model_path}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MLP(input_dim=X_test.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model(X_test_tensor).squeeze().cpu().numpy()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pcc, _ = pearsonr(y_test, y_pred)

    print({
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "PCC": pcc,
        "Best Weights": model_path
    })


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train models to predict pIC50 from molecular data.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--test_ML", type=str, help="Path to the trained ML model.")
    parser.add_argument("--test_deep", type=str, help="Path to the trained Deeplearning model.")
    
    args = parser.parse_args()

    save_path = "weights/" + (args.file_path).split('/')[-1].split('.')[0]
    os.makedirs(save_path, exist_ok=True)

    print(f"...Loading data from {args.file_path}")    
    X_train, X_test, y_train, y_test = load_data(args.file_path)
    
    # print(">>> Training ML Models...")
    # train_ml_models(X_train, y_train, X_test, y_test, save_path)
    # print()

    # print(">>> Training Deep Learning Model...")
    # train_deep_learning(X_train, y_train, X_test, y_test, save_path)

    test_deeplearning(X_test, y_test, "./weights/ic50_augmented_20k/best_deep_weight.pth")
