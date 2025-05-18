import os
import time
import torch

from utils.logger import Logger, set_log, start_log, train_log, check_log
from utils.loader import load_ckpt, load_model_from_ckpt, load_data, \
                         load_seed, load_device, load_sampling_fn
from utils.graph_utils import init_flags, quantize_mol
from utils.mol_utils import gen_mol, mols_to_smiles

from utils.regressor_utils import load_regressor_from_ckpt, load_regressor_ckpt
from utils.regressor_utils import train_log as train_log_prop
from utils.regressor_utils import sample_log as sample_log_prop

from _evaluate import evaluate

class Sampler(object):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.device = load_device(self.config.gpu)

    def check_config(self, config1, config2):
        assert config1.data.batch_size == config2.data.batch_size, 'Batch size Mismatch'
        assert config1.data.max_node_num == config2.data.max_node_num, 'Max node num Mismatch'
        assert config1.data.max_feat_num == config2.data.max_feat_num, 'Max feat. num Mismatch'
        assert config1.sde.x == config2.sde.x, 'SDE Mismatch: X'
        assert config1.sde.adj == config2.sde.adj, 'SDE Mismatch: Adj'
    
    def sample(self, n_samples_to_generate):

        self.config.sample.num_samples = n_samples_to_generate
        
        self.diff = load_ckpt(self.config, self.device)['diff']
        self.config_diff = self.diff['config']

        self.prop = load_regressor_ckpt(self.config, self.device)['prop']
        self.config_prop = self.prop['config']

        self.check_config(self.config_diff, self.config_prop)
        load_seed(self.config.seed)

        self.log_folder_name, self.log_dir, _ = set_log(self.config_diff, foldername=self.config_prop.train.prop, is_train=False)
        self.log_name = f"{self.config.model.diff.ckpt}_{self.config.model.prop.ckpt}"
        print(f'logname: {self.log_name}')
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            start_log(logger, self.config_diff)
            train_log(logger, self.config_diff)
            train_log_prop(logger, self.config_prop)
            sample_log_prop(logger, self.config.model.prop)
        
        self.model_x = load_model_from_ckpt(self.diff['params_x'], self.diff['x_state_dict'], self.device)
        self.model_adj = load_model_from_ckpt(self.diff['params_adj'], self.diff['adj_state_dict'], self.device)

        self.regressor = load_regressor_from_ckpt(self.prop['params'], self.prop['state_dict'], self.device)
        self.regressor.eval()
        self.sampling_fn = load_sampling_fn(self.config_diff, self.config.model.diff, self.config.model.prop,
                                            self.config.sample, self.device)
        self.config_prop.data.dir = self.config.data.dir
        self.train_graph_list, _ = load_data(self.config_prop, get_graph_list=True)     # for init_flags
        self.init_flags = init_flags(self.train_graph_list, self.config_diff, n_samples_to_generate).to(self.device[0])

        x, adj = self.sampling_fn(self.model_x, self.model_adj, self.init_flags, self.regressor)

        samples_int = quantize_mol(adj)

        samples_int_ = samples_int - 1
        samples_int_[samples_int_ == -1] = 3

        adj = torch.nn.functional.one_hot(torch.tensor(samples_int_), num_classes=4).permute(0,3,1,2)
        x = torch.where(x > 0.5, 1, 0)
        x = torch.cat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)

        gen_mols, _ = gen_mol(x, adj, self.config_diff.data.data)
        gen_smiles = mols_to_smiles(gen_mols)

        return gen_smiles
