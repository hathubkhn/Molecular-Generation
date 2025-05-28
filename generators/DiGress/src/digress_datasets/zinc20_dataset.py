from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
import pandas as pd

import utils
from analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges, compute_molecular_metrics
from digress_datasets.abstract_dataset import AbstractDatasetInfos, MolecularDataModule


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]

def read_txt(fpath):
    smiles_list = []
    with open(fpath, 'r') as file:
        lines = file.readlines()
    for line in lines:
        smile = line.split()[0]
        smiles_list.append(smile)
    return smiles_list

atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H', 'P', 'Si', 'I']

class ZINCDataset(InMemoryDataset):
    def __init__(self, stage, root, filter_dataset=True, transform=None, pre_transform=None, pre_filter=None):
        self.stage = stage
        self.atom_decoder = atom_decoder
        self.filter_dataset = filter_dataset
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['zinc00_train_medium.txt', 'zinc00_val_medium.txt', 'new_test.txt']

    @property
    def split_file_name(self):
        return ['zinc00_train_medium.txt', 'zinc00_val_medium.txt', 'new_test.txt']
    
    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.filter_dataset:
            return ['train_filtered.pt', 'test_filtered.pt', 'test_scaffold_filtered.pt']
        else:
            return ['train.pt', 'test.pt', 'test_scaffold.pt']


    def process(self):
        RDLogger.DisableLog('rdApp.*')
        types = {atom: i for i, atom in enumerate(self.atom_decoder)}

        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        
        
        path = self.split_paths[self.file_idx]
        smiles_list = read_txt(path)

        data_list = []
        smiles_kept = []

        for i, smile in enumerate(tqdm(smiles_list)):
            mol = Chem.MolFromSmiles(smile)
            N = mol.GetNumAtoms()

            check = 0

            type_idx = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() in types:
                    type_idx.append(types[atom.GetSymbol()])
                else:
                    check = 1
                    break

            if check == 1:
                continue

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            if len(row) == 0:
                continue

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            y = torch.zeros(size=(1, 0), dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

            if self.filter_dataset:
                # Try to build the molecule again from the graph. If it fails, do not add it to the training set
                dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
                dense_data = dense_data.mask(node_mask, collapse=True)
                X, E = dense_data.X, dense_data.E

                assert X.size(0) == 1
                atom_types = X[0]
                edge_types = E[0]
                mol = build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder)
                smiles = mol2smiles(mol)
                if smiles is not None:
                    try:
                        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                        if len(mol_frags) == 1:
                            data_list.append(data)
                            smiles_kept.append(smiles)

                    except Chem.rdchem.AtomValenceException:
                        print("Valence error in GetmolFrags")
                    except Chem.rdchem.KekulizeException:
                        print("Can't kekulize molecule")
            else:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

        if self.filter_dataset:
            smiles_save_path = osp.join(pathlib.Path(self.raw_paths[0]).parent, f'new_{self.stage}.smiles')
            print(smiles_save_path)
            with open(smiles_save_path, 'w') as f:
                f.writelines('%s\n' % s for s in smiles_kept)
            print(f"Number of molecules kept: {len(smiles_kept)} / {len(smiles_list)}")



class ZINCDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.remove_h = False
        self.datadir = cfg.dataset.datadir
        self.filter_dataset = cfg.dataset.filter
        self.train_smiles = []
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': ZINCDataset(stage='train', root=root_path, filter_dataset=self.filter_dataset),
                    'val': ZINCDataset(stage='val', root=root_path, filter_dataset=self.filter_dataset),
                    'test': ZINCDataset(stage='test', root=root_path, filter_dataset=self.filter_dataset)}
        super().__init__(cfg, datasets)




class ZINCinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False, meta=None):        
        self.name = 'zin20'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = False

        self.atom_decoder = atom_decoder
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        #############      ['C',   'N',   'S',   'O',   'F',   'Cl',    'Br',    'H',  'P',      'Si',      'I']
        self.atom_weights = {0: 12, 1: 14, 2: 32, 3: 16, 4: 19, 5: 35.4, 6: 79.9, 7: 1, 8: 30.97, 9: 28.09, 10: 126.9}
        self.valencies =    [4,     3,     4,     2,     1,     1,       1,       1,    3,        4,        1]
        
        self.num_atom_types = len(self.atom_decoder)
        self.max_weight = 350

        meta_files = dict(n_nodes=f'{self.name}_n_counts.txt',
                          node_types=f'{self.name}_atom_types.txt',
                          edge_types=f'{self.name}_edge_types.txt',
                          valency_distribution=f'{self.name}_valencies.txt')
    
        self.n_nodes = torch.tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.3816e-05, 2.8579e-05,
                                    4.6917e-04, 3.6772e-03, 1.8633e-02, 6.8308e-02, 1.2460e-01, 1.5613e-01,
                                    8.2807e-02, 4.3130e-02, 6.9251e-02, 1.1238e-01, 1.5354e-01, 1.2288e-01,
                                    4.4109e-02, 2.8579e-05])
        # self.n_nodes = None
        self.max_n_nodes = len(self.n_nodes) - 1 if self.n_nodes is not None else None
        self.max_n_nodes = self.n_nodes.size(0) - 1 if self.n_nodes is not None else None
        self.node_types = torch.tensor([7.7900e-01, 9.9487e-02, 1.1708e-02, 7.8005e-02, 2.3163e-02, 7.8259e-03,
        7.4722e-04, 3.5668e-05, 7.4944e-06, 1.5128e-05, 4.1635e-07])
        self.edge_types = torch.tensor([8.8663e-01, 6.7779e-02, 4.2474e-03, 8.7829e-06, 4.1338e-02])
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[:7] = torch.tensor([0.0, 1.525318771600723267e-01, 3.088481128215789795e-01, 3.398384451866149902e-01, 1.976608633995056152e-01, 5.146131734363734722e-04, 6.060721934773027897e-04])

        if meta is None:
            meta = dict(n_nodes=None, node_types=None, edge_types=None, valency_distribution=None)
        assert set(meta.keys()) == set(meta_files.keys())
        for k, v in meta_files.items():
            if (k not in meta or meta[k] is None) and os.path.exists(v):
                meta[k] = np.loadtxt(v)
                setattr(self, k, meta[k])
        if recompute_statistics or self.n_nodes is None:
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt(meta_files["n_nodes"], self.n_nodes.numpy())
            self.max_n_nodes = len(self.n_nodes) - 1
        if recompute_statistics or self.node_types is None:
            self.node_types = datamodule.node_types()                                     # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt(meta_files["node_types"], self.node_types.numpy())

        if recompute_statistics or self.edge_types is None:
            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt(meta_files["edge_types"], self.edge_types.numpy())
        if recompute_statistics or self.valency_distribution is None:
            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt(meta_files["valency_distribution"], valencies.numpy())
            self.valency_distribution = valencies
        # after we can be sure we have the data, complete infos
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)


def get_train_smiles(cfg, datamodule, dataset_infos, evaluate_dataset=False):
    base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_path = os.path.join(base_path, cfg.dataset.datadir)

    train_smiles = None
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.array(open(smiles_path).readlines())

    if evaluate_dataset:
        train_dataloader = datamodule.dataloaders['train']
        all_molecules = []
        for i, data in enumerate(tqdm(train_dataloader)):
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return train_smiles


if __name__ == "__main__":
    ds = [ZINCDataset(s, os.path.join(os.path.abspath(__file__), "../../../../../../data/generator/digress/zinc20"), preprocess=True) for s in ["train", "val", "test"]]