import argparse
from omegaconf import DictConfig
import hydra
import gradio as gr
import cv2
import os
from PIL import Image
import numpy as np
import base64
import io
import random
import pandas as pd
from filterer import SMILESFilterer
import sys
from hydra import initialize, compose
from omegaconf import DictConfig
import time

def load_digress_config() -> DictConfig:
    with initialize(version_base="1.3", config_path="./configs/digress"):
        cfg = compose(config_name="config")
    return cfg

def get_final_smiles(generator, filterer, thresholds, n_final_smiles, scale_up=20):
    final_smiles = []
    count = 0

    lower_logP, upper_logP, lower_SA, upper_SA, lower_pic50, upper_pic50, max_rings_count = thresholds

    while len(final_smiles) < n_final_smiles:
        start = time.time()
        print(f"Generate batch {count}...")
        count += 1
        generated_smiles = generator.generate(n_samples=n_final_smiles*scale_up)
        filtered_smiles = filterer.filter_lst_smiles(generated_smiles, lower_logP=lower_logP, upper_logP=upper_logP, lower_SA=lower_SA, upper_SA=upper_SA, lower_pic50=lower_pic50, upper_pic50=upper_pic50, max_rings_count=max_rings_count)

        final_smiles.extend(filtered_smiles)
        print(f"Batch finished in {time.time() - start}")
    
    return final_smiles[:n_final_smiles]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and filter SMILES")
    parser.add_argument("--model", type=str, default="digress", choices=["digress", "mood", "gdss", "vae"], help="Choose SMILES generator")
    parser.add_argument("--n_final_smiles", type=str, default=20, help="Number of SMILES to save")
    args = parser.parse_args()

    if args.model == "digress":
        sys.path.append("./generators/DiGress/")
        sys.path.append("./generators/DiGress/src/")
        digress_cfg = load_digress_config()
        from generator import DigressGenerator
        generator = DigressGenerator(digress_cfg)
    
    elif args.model == "mood":
        sys.path.append("./generators/MOOD/")
        from generator import MOODGenerator
        generator = MOODGenerator(cfg_dir="./configs/mood")
        
    elif args.model == "gdss":
        sys.path.append("./generators/GDSS/")
        from generator import GDSSGenerator
        generator = GDSSGenerator(cfg_dir="./configs/gdss")
        
    elif args.model == "vae":
        sys.path.append("./generators/Molecular-VAE/")
        from generator import VAEGenerator
        generator = VAEGenerator(cfg_dir="./configs/vae")
        
    else:
        raise ValueError("Model not implemented")

    filterer = SMILESFilterer()
    final_smiles = get_final_smiles(generator, filterer, args.n_final_smiles)
    
    print(final_smiles)