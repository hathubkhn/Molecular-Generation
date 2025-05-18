import os.path as osp
import numpy as np
import json
import h5py

def one_hot_array(i, n):
    return map(int, [ix == i for ix in range(n)])

def one_hot_index(vec, charset):
    return map(charset.index, vec)

def from_one_hot_array(vec):
    oh = np.where(vec == 1)
    if oh[0].shape == (0, ):
        return None
    return int(oh[0][0])

def decode_smiles_from_indexes(vec, charset):
    decoded_chars = []
    for index in vec:
        if index is None:
            break
        decoded_chars.append(charset[index])
    decoded_smiles = "".join(decoded_chars).strip()
    return decoded_smiles
    
def decode_smiles_from_indexes_(vec, charset):
    decoded_chars = [charset[index] for index in vec]
    string_list = [b.decode('utf-8') for b in decoded_chars]
    decoded_smiles = "".join(string_list).strip()
    return decoded_smiles

def load_custom_dataset(data_dir):
    train_fpath = osp.join(data_dir, "train.npy")
    val_fpath = osp.join(data_dir, "val.npy")

    print("Loading training data...")
    data_train = np.load(train_fpath)
    print("Loading validation data...")
    data_val = np.load(val_fpath)
    print("Loading charset...")
    with open(osp.join(data_dir, "charset.json"), "r") as jsonf:
        charset = json.load(jsonf)

    return data_train, data_val, charset

def load_dataset(filename, split = True):
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset =  h5f['charset'][:]
    h5f.close()
    if split:
        return (data_train, data_test, charset)
    else:
        return (data_test, charset)