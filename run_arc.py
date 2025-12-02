import os
import json
import torch
import random
import numpy as np
import shutil
from pathlib import Path
from torch.utils.data import Dataset
from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D
from tiny_recursive_model.trm import TinyRecursiveModel
from tiny_recursive_model.trainer import Trainer

# --- 1. DATASET DOWNLOADER ---
def download_arc_data():
    data_path = Path("ARC-AGI/data/training")
    if data_path.exists():
        print("ARC-AGI dataset found locally.")
        return

    print("Downloading ARC-AGI Dataset from GitHub...")
    # We clone the specific folder needed
    if not Path("ARC-AGI").exists():
        os.system("git clone https://github.com/fchollet/ARC-AGI.git")
    
    if not data_path.exists():
        print("Error: Download failed. Please run 'git clone https://github.com/fchollet/ARC-AGI.git' manually.")
        exit()

# --- 2. DATA AUGMENTATION (CRITICAL FOR ARC) ---
# ARC has only 400 training tasks. We must augment them to train a neural net.
def augment_grid(grid):
    # Random Rotation (0, 90, 180, 270)
    k = random.randint(0, 3)
    grid = np.rot90(grid, k)
    # Random Flip
    if random.random() > 0.5:
        grid = np.flipud(grid)
    return grid.copy()

class ARCDataset(Dataset):
    def __init__(self, root_dir="ARC-AGI/data/training", augment=True, limit=None):
        self.root_dir = Path(root_dir)
        self.augment = augment
        self.pairs = []
        
        # Load all JSON tasks
        files = list(self.root_dir.glob("*.json"))
        if limit: files = files[:limit]
        
        print(f"Loading {len(files)} ARC Tasks...")
        
        for fpath in files:
            with open(fpath, 'r') as f:
                task = json.load(f)
                # ARC tasks have multiple training pairs per file. We load them all.
                for pair in task['train']:
                    inp = np.array(pair['input'])
                    out = np.array(pair['output'])
                    self.pairs.append((inp, out))
        
        print(f"Loaded {len(self.pairs)} training pairs.")

    def __len__(self): 
        # We pretend the dataset is larger to allow dynamic augmentation per epoch
        return len(self.pairs) * 10 

    def __getitem__(self, idx):
        # Map virtual index to real pair
        real_idx = idx % len(self.pairs)
        inp_grid, out_grid = self.pairs[real_idx]
        
        # Augment
        if self.augment:
            inp_grid = augment_grid(inp_grid)
            out_grid = augment_grid(out_grid)
        
        # Flatten (2D -> 1D Sequence)
        inp_flat = torch.tensor(inp_grid.flatten()).long()
        out_flat = torch.tensor(out_grid.flatten()).long()
        
        # Padding Strategy
        # ARC grids are max 30x30 = 900 tokens.
        MAX_LEN = 900
        
        pad_in = MAX_LEN - len(inp_flat)
        pad_out = MAX_LEN - len(out_flat)
        
        # Safety check for weirdly large grids
        if pad_in < 0: inp_flat = inp_flat[:MAX_LEN]; pad_in = 0
        if pad_out < 0: out_flat = out_flat[:MAX_LEN]; pad_out = 0

        # Pad with 10 (Color 10 = Padding/Empty)
        inp_final = torch.cat([inp_flat, torch.full((pad_in,), 10).long()])
        # Target Padding is -100 (Ignore)
        out_final = torch.cat([out_flat, torch.full((pad_out,), -100).long()])
        
        return inp_final, out_final

# --- 3. BENCHMARK RUNNER ---
def run_arc_benchmark():
    print("--- BENCHMARK: ARC-AGI (Reasoning) ---")
    
    download_arc_data()
    
    # 1. Dataset
    dataset = ARCDataset()
    
    # 2. Model Configuration
    # ARC requires high capacity.
    model = TinyRecursiveModel(
        dim = 512,              # Wide network for complex logic
        num_tokens = 11,        # 0-9 Colors + 10 Padding
        max_seq_len = 1024,     # Fits 30x30 grid
        network = MLPMixer1D(
            dim = 512, 
            depth = 8,          # Deeper mixer
            seq_len = 900 
        ),
        num_refinement_blocks = 4, 
        num_latent_refinements = 6 
    )

    # 3. Trainer
    trainer = Trainer(
        model,
        dataset,
        learning_rate = 1e-4,   # Lower LR for delicate visual logic
        weight_decay = 0.0,     # Keep memory
        batch_size = 16,        # Smaller batch due to 512 dim
        epochs = 100,           # ARC needs long training
        max_recurrent_steps = 16,
        warmup_steps = 1000,
        compile_model = False,
        cpu = not torch.cuda.is_available()
    )

    # 4. Train
    print("Starting Training (This simulates the 'Pre-training' phase on ARC)...")
    trainer.forward()
    
    # Note: True ARC evaluation requires "Few-Shot Test-Time Adaptation" 
    # (finetuning on the specific test example). 
    # This script validates that the model can learn the *general physics* of ARC.

if __name__ == "__main__":
    run_arc_benchmark()