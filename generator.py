import os
import os.path as osp
import sys
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import initialize, compose
from omegaconf import DictConfig
from tqdm import tqdm
import yaml
from easydict import EasyDict as edict
import time
import argparse
import sys
import json
import numpy as np
import torch.optim as optim

def load_digress_config() -> DictConfig:
    with initialize(version_base="1.3", config_path="./configs/digress"):
        cfg = compose(config_name="config")
    return cfg


class VAEGenerator():
    def __init__(self, cfg_dir):
        self.cfg = edict(yaml.load(open(osp.join(cfg_dir, "vae.yaml"), "r"), Loader=yaml.FullLoader))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        from vae import MolecularVAE
        charset = self.load_charset(self.cfg.data_dir)
        self.model = MolecularVAE(charset, self.cfg.max_molecule_len, self.cfg.latent_dim, self.device)

    @staticmethod
    def load_charset(data_dir):
        with open(osp.join(data_dir, "charset.json"), "r") as jsonf:
            charset = json.load(jsonf)
        return charset

    @staticmethod
    def load_custom_dataset(data_dir):
        train_fpath = osp.join(data_dir, "train.npy")
        val_fpath = osp.join(data_dir, "val.npy")
    
        data_train = np.load(train_fpath)
        data_val = np.load(val_fpath)
    
        return data_train, data_val
        
    def train(self):
        data_train, data_test = self.load_custom_dataset(self.cfg.data_dir)
        data_train = torch.utils.data.TensorDataset(torch.from_numpy(data_train))
        train_loader = torch.utils.data.DataLoader(data_train, batch_size=self.cfg.batch_size, shuffle=True)
    
        data_test = torch.utils.data.TensorDataset(torch.from_numpy(data_test))
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=self.cfg.batch_size, shuffle=False)
        
        self.model = self.model.to(self.device)
    
        if self.cfg.resume:
            state_dict = torch.load(self.cfg.resume)
            self.model.load_state_dict(state_dict)
            print(f"Resume training from weight {self.cfg.resume}")
        
        print(">>>> START TRAINING...")
        optimizer = optim.Adam(self.model.parameters())
        best_val_loss = float("inf")
        for epoch in range(1, self.cfg.n_epochs + 1):
            self.model, optimizer = self.model.train_one_epoch(self.model, optimizer, train_loader, epoch)
            if epoch % 1 == 0:
                val_loss = self.model.evaluate(self.model, test_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), f"./weights/vae_{best_val_loss}.pth")

        print(">>>> FINISHED TRAINING!")

    def generate(self, n_samples=10):
        state_dict = torch.load(self.cfg.test_only)
        self.model.load_state_dict(state_dict)
        print(f"Loaded weight from {self.cfg.test_only}")

        self.model = self.model.to(self.device)
        
        print(f"Generating {n_samples} samples")
        with torch.no_grad():
            z = torch.randn(n_samples, self.model.latent_dim).to(self.device)
            generated = self.model.decode(z)

            smiles_list = []
            for idx in tqdm(range(n_samples)):
                sampled = generated[idx].cpu().numpy().reshape(1, self.model.max_molecule_len, len(self.model.charset)).argmax(axis=2)[0]
                decoded_smiles = self.model.decode_smiles_from_indexes(sampled, self.model.charset)
                smiles_list.append(decoded_smiles)

        return smiles_list


class MOODGenerator():
    def __init__(self, cfg_dir):
        self.train_cfg = edict(yaml.load(open(osp.join(cfg_dir, "prop_train.yaml"), "r"), Loader=yaml.FullLoader))
        self.train_cfg.seed = 42
        self.train_cfg.gpu = 0

        self.sample_cfg = edict(yaml.load(open(osp.join(cfg_dir, "sample.yaml"), "r"), Loader=yaml.FullLoader))
        self.sample_cfg.seed = 42
        self.sample_cfg.gpu = 0
        
        from prop_trainer import Trainer
        self.trainer = Trainer(self.train_cfg)
        from sampler_infer import Sampler
        self.sampler = Sampler(self.sample_cfg)

    def train(self):
        print(">>>> START TRAINING...")
        self.trainer.train()
        print(">>>> FINISHED TRAINING!")

    def generate(self, n_samples=10):
        print(f"Generating {n_samples} samples...")
        return self.sampler.sample(n_samples)


class GDSSGenerator():
    def __init__(self, cfg_dir):
        self.train_cfg = edict(yaml.load(open(osp.join(cfg_dir, "zinc250k.yaml"), "r"), Loader=yaml.FullLoader))
        self.train_cfg.seed = 42

        self.sample_cfg = edict(yaml.load(open(osp.join(cfg_dir, "sample_zinc250k.yaml"), "r"), Loader=yaml.FullLoader))
        self.sample_cfg.seed = 42

        from trainer import Trainer
        self.trainer = Trainer(self.train_cfg)
        from sampler_infer_gdss import Sampler_mol
        self.sampler = Sampler_mol(self.sample_cfg)
        
    def train(self):
        print(">>>> START TRAINING...")
        ts = time.strftime("%b%d-%H:%M:%S", time.gmtime())
        self.trainer.train(ts)
        print(">>>> FINISHED TRAINING!")

    def generate(self, n_samples=10):
        print(f"Generating {n_samples} samples...")
        return self.sampler.sample(n_samples)


class DigressGenerator():
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        dataset_config = cfg["dataset"]
        if dataset_config["name"] != "zinc20":
            raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

        from diffusion_model_discrete import DiscreteDenoisingDiffusion
        from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
        import utils
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization
    
        from datasets import zinc20_dataset
        datamodule = zinc20_dataset.ZINCDataModule(cfg)
        dataset_infos = zinc20_dataset.ZINCinfos(datamodule, cfg)
        train_smiles = None
    
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    
        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)
    
        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)
    
        model_kwargs = {"dataset_infos": dataset_infos, "train_metrics": train_metrics,
                        "sampling_metrics": sampling_metrics, "visualization_tools": visualization_tools,
                        "extra_features": extra_features, "domain_features": domain_features}
                
        self.datamodule = datamodule
        self.model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.cfg.general.test_only:
            checkpoint = torch.load(self.cfg.general.test_only, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded weight from {self.cfg.general.test_only}")
    
    def train(self):
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint

        callbacks = []
        if self.cfg.train.save_model:
            checkpoint_callback = ModelCheckpoint(dirpath=f"weights/{self.cfg.general.name}",
                                              filename="{epoch}",
                                              monitor="val/epoch_NLL",
                                              save_top_k=5,
                                              mode="min",
                                              every_n_epochs=1)
            last_ckpt_save = ModelCheckpoint(dirpath=f"weights/{self.cfg.general.name}", filename="last", every_n_epochs=1)
            callbacks.append(last_ckpt_save)
            callbacks.append(checkpoint_callback)
    
        if self.cfg.train.ema_decay > 0:
            ema_callback = utils.EMA(decay=self.cfg.train.ema_decay)
            callbacks.append(ema_callback)

        use_gpu = self.cfg.general.gpus > 0 and torch.cuda.is_available()
        trainer = Trainer(gradient_clip_val=self.cfg.train.clip_grad,
                          strategy="ddp_find_unused_parameters_true",
                          accelerator="gpu" if use_gpu else "cpu",
                          devices=self.cfg.general.gpus if use_gpu else 1,
                          max_epochs=self.cfg.train.n_epochs,
                          check_val_every_n_epoch=self.cfg.general.check_val_every_n_epochs,
                          fast_dev_run=self.cfg.general.name == "debug",
                          enable_progress_bar=True,
                          callbacks=callbacks,
                          log_every_n_steps=50,
                          logger = [])
    
        print(">>>> START TRAINING...")
        trainer.fit(self.model, datamodule=self.datamodule, ckpt_path=self.cfg.general.resume)
        print(">>>> FINISHED TRAINING!")
        trainer.test(self.model, datamodule=self.datamodule)
            
    def generate(self, n_samples=10):
        import os
        os.environ["WANDB_DISABLED"] = "true"
        print(f"Generating {n_samples} samples...")
        smiles = self.model.get_smiles(n_samples, gen_batchsize=4096)
        return smiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or generate SMILES with Generators")
    parser.add_argument("--model", type=str, required=True, choices=["digress", "mood", "gdss", "vae"], default="digress", help="Choose generative model")
    parser.add_argument("--task", type=str, choices=["train", "generate"], default="generate")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training.")
    parser.add_argument("--test_only", type=str, default="")
    parser.add_argument("--n_samples_to_generate", type=int, default=20)
    args = parser.parse_args()

    if args.model == "digress":
        sys.path.append("./generators/DiGress/")
        sys.path.append("./generators/DiGress/src/")
        digress_cfg = load_digress_config()
        digress_cfg.train.n_epochs = args.n_epochs
        digress_cfg.train.batch_size = args.batch_size
        if args.test_only:
            digress_cfg.general.test_only = args.test_only
            digress_cfg.train.batch_size = 2048
        generator = DigressGenerator(digress_cfg)
    
    elif args.model == "mood":
        sys.path.append("./generators/MOOD/")
        generator = MOODGenerator(cfg_dir="./configs/mood")
        generator.train_cfg.train.num_epochs = args.n_epochs
        generator.train_cfg.data.batch_size = args.batch_size
        if args.test_only:
            ckpt = args.test_only.split(".")[0]
            generator.sample_cfg.model.prop = ckpt
        
    elif args.model == "gdss":
        sys.path.append("./generators/GDSS/")
        generator = GDSSGenerator(cfg_dir="./configs/gdss")
        generator.train_cfg.train.num_epochs = args.n_epochs
        generator.train_cfg.data.batch_size = args.batch_size
        if args.test_only:
            ckpt = args.test_only.split(".")[0]
            generator.sample_cfg.ckpt = ckpt
        
    elif args.model == "vae":
        sys.path.append("./generators/Molecular-VAE/")
        generator = VAEGenerator(cfg_dir="./configs/vae")
        generator.cfg.n_epochs = args.n_epochs
        generator.cfg.batch_size = args.batch_size
        if args.test_only:
            generator.cfg.test_only = args.test_only

    else:
        raise ValueError("Model not implemented")
    
    
    if args.task == "train":
        generator.train()
    else:
        generated_smiles = generator.generate(n_samples=args.n_samples_to_generate)
        lines = "\n".join(generated_smiles)
        with open(f"generated_smiles_{args.model}.txt", "w") as output_f:
            output_f.writelines(lines)
        print(f"Saved {args.n_samples_to_generate} samples to generated_smiles_{args.model}.txt")