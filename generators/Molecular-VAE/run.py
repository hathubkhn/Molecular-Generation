from tqdm import tqdm
from vae import MolecularVAE
from utils import *
import numpy as np
import argparse
import torch.optim as optim
import torch
import torch.nn.functional as F

def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
    xent_loss = F.binary_cross_entropy(x_decoded_mean, x, size_average=False)
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return xent_loss + kl_loss

def train(epoch):
    model.train()
    train_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

    for batch_idx, (data,) in progress_bar:
        data = data.to(device)
        optimizer.zero_grad()
        output, mean, logvar = model(data)

        if batch_idx==0:
              inp = data.cpu().numpy()
              outp = output.cpu().detach().numpy()
              lab = data.cpu().numpy()
              print("Input:")
              print(decode_smiles_from_indexes(map(from_one_hot_array, inp[0]), charset))
              print("Label:")
              print(decode_smiles_from_indexes(map(from_one_hot_array, lab[0]), charset))
              sampled = outp[0].reshape(1, model.max_molecule_len, len(charset)).argmax(axis=2)[0]
              print("Output:")
              print(decode_smiles_from_indexes(sampled, charset))
        
        loss = vae_loss(output, data, mean, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Cập nhật tiến trình hiển thị
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = train_loss / len(train_loader.dataset)
    print(f'Train Loss: {avg_loss:.6f}')
    return avg_loss

def evaluate():
    model.eval()
    val_loss = 0
    progress_bar = tqdm(test_loader, total=len(test_loader), desc="Evaluating")

    with torch.no_grad():
        for data, in progress_bar:
            data = data.to(device)
            output, mean, logvar = model(data)
            loss = vae_loss(output, data, mean, logvar).item()
            val_loss += loss

            # Cập nhật thanh tiến trình
            progress_bar.set_postfix(loss=loss)

    val_loss /= len(test_loader.dataset)
    print(f'Validation Loss: {val_loss:.6f}')
    return val_loss



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load dataset and set training parameters.")
    parser.add_argument("--data_dir", type=str, default="data/zinc20", 
                        help="Path to the dataset directory.")
    parser.add_argument("--n_epochs", type=int, default=1000, 
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=2048, 
                        help="Batch size for training.")
    # parser.add_argument("--test_only", type=str, default="./weights/molecular_vae_40.084521875.pth")
    parser.add_argument("--test_only", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--n_samples_to_generate", type=int, default=5000)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_train, data_test, charset = load_custom_dataset(args.data_dir)
    # data_train, data_test, charset = load_dataset('./data/processed.h5')

    # print("Charset: ", charset)
    
    data_train = torch.utils.data.TensorDataset(torch.from_numpy(data_train))
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    
    data_test = torch.utils.data.TensorDataset(torch.from_numpy(data_test))
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False)
    
    model = MolecularVAE(charset, 80, 292, device).to(device)

    if args.resume != "":
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict)
        print(f"Resume training from weight {args.resume}")
    
    if args.test_only == "":
        print(">>>> START TRAINING...")
        optimizer = optim.Adam(model.parameters())
        best_val_loss = float('inf')
        for epoch in range(1, args.n_epochs):
            train_loss = train(epoch)
            if epoch % 5 == 0:
                val_loss = evaluate()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f'weights/molecular_vae_{best_val_loss}.pth')
                    print('Model saved!')
    else:
        state_dict = torch.load(args.test_only)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {args.test_only}")

    print(">>>> Generating SMILES...")
    generated_smiles = model.generate(args.n_samples_to_generate)

    lines = [smiles + "\n" for smiles in generated_smiles]
    with open("final_smiles.txt", "w") as output_f:
        output_f.writelines(lines)

    print("Save generated SMILES to ./final_smiles.txt")