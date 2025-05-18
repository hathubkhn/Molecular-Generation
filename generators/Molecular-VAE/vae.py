import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# from utils import *

class MolecularVAE(nn.Module):
    def __init__(self, charset, max_molecule_len, latent_dim, device):
        super(MolecularVAE, self).__init__()

        self.device = device

        self.latent_dim = latent_dim
        self.max_molecule_len = max_molecule_len
    
        self.conv_1 = nn.Conv1d(self.max_molecule_len, 9, kernel_size=9)
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)

        final_conv_size = 10 * (len(charset) - 26)
        self.linear_0 = nn.Linear(final_conv_size, 435)
        # self.linear_0 = nn.Linear(70, 435)
        self.linear_1 = nn.Linear(435, self.latent_dim)
        self.linear_2 = nn.Linear(435, self.latent_dim)

        self.linear_3 = nn.Linear(self.latent_dim, self.latent_dim)
        self.gru = nn.GRU(self.latent_dim, 501, 3, batch_first=True)
        self.linear_4 = nn.Linear(501, len(charset))
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.charset = charset

    @staticmethod
    def one_hot_array(i, n):
        return map(int, [ix == i for ix in range(n)])

    @staticmethod
    def one_hot_index(vec, charset):
        return map(charset.index, vec)

    @staticmethod
    def from_one_hot_array(vec):
        oh = np.where(vec == 1)
        if oh[0].shape == (0, ):
            return None
        return int(oh[0][0])

    @staticmethod
    def decode_smiles_from_indexes(vec, charset):
        decoded_chars = []
        for index in vec:
            if index is None:
                break
            decoded_chars.append(charset[index])
        decoded_smiles = "".join(decoded_chars).strip()
        return decoded_smiles

    def encode(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))
        return self.linear_1(x), self.linear_2(x)

    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z):
        z = F.selu(self.linear_3(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.max_molecule_len, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.linear_4(out_reshape), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar

    @staticmethod
    def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
        xent_loss = F.binary_cross_entropy(x_decoded_mean, x, size_average=False)
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        return xent_loss + kl_loss
    
    def evaluate(self, model, test_loader):
        model.eval()
        val_loss = 0
        progress_bar = tqdm(test_loader, total=len(test_loader), desc="Evaluating")
    
        with torch.no_grad():
            for data, in progress_bar:
                data = data.to(self.device)
                output, mean, logvar = model(data)
                loss = self.vae_loss(output, data, mean, logvar).item()
                val_loss += loss
    
                progress_bar.set_postfix(loss=loss)
    
        val_loss /= len(test_loader.dataset)
        print(f'Validation Loss: {val_loss:.6f}')
        return val_loss
    
    def train_one_epoch(self, model, optimizer, train_loader, epoch, print_sample=False):
        model.train()
        train_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    
        for batch_idx, (data,) in progress_bar:
            data = data.to(self.device)
            optimizer.zero_grad()
            output, mean, logvar = model(data)
       
            if batch_idx==0:
                  inp = data.cpu().numpy()
                  outp = output.cpu().detach().numpy()
                  lab = data.cpu().numpy()
                  if print_sample:
                      print("Input:")
                      print(self.decode_smiles_from_indexes(map(from_one_hot_array, inp[0]), self.charset))
                      print("Label:")
                      print(self.decode_smiles_from_indexes(map(from_one_hot_array, lab[0]), self.charset))
                      sampled = outp[0].reshape(1, model.max_molecule_len, len(self.charset)).argmax(axis=2)[0]
                      print("Output:")
                      print(self.decode_smiles_from_indexes(sampled, self.charset))
            
            loss = self.vae_loss(output, data, mean, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
            progress_bar.set_postfix(loss=loss.item())
    
        avg_loss = train_loss / len(train_loader.dataset)
        print(f'Train Loss: {avg_loss:.6f}')
        return model, optimizer