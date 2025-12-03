import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D
from tiny_recursive_model.trm import TinyRecursiveModel
from tiny_recursive_model.trainer import Trainer

try:
    from datasets import load_dataset
    HAS_HF = True
except ImportError:
    HAS_HF = False

class Sudoku1MDataset(Dataset):
    def __init__(self, split="train", limit=10000):
        self.data = []
        print(f"Loading {limit} puzzles...")

        loaded = False
        
        # STRATEGY 1: Official Ritvik19 Dataset (Active)
        if HAS_HF and not loaded:
            try:
                print("Attempting to load 'Ritvik19/Sudoku-Dataset'...")
                # This dataset is huge (17M), streaming is required
                ds = load_dataset("Ritvik19/Sudoku-Dataset", split="train", streaming=True)
                
                count = 0
                for item in ds:
                    if count >= limit: break
                    
                    # Columns are 'puzzle' and 'solution' in this dataset
                    quiz_str = item['puzzle']
                    sol_str = item['solution']
                    
                    quiz = np.array([int(c) for c in quiz_str])
                    solution = np.array([int(c) for c in sol_str])
                    
                    self.data.append((torch.tensor(quiz).long(), torch.tensor(solution).long()))
                    count += 1
                loaded = True
            except Exception as e:
                print(f"HF Load failed: {e}")

        # STRATEGY 2: Local CSV (Manual Fallback)
        if not loaded:
            print("HF failed. Checking for local 'sudoku.csv'...")
            if os.path.exists("sudoku.csv") and os.path.getsize("sudoku.csv") > 0:
                try:
                    df = pd.read_csv("sudoku.csv", nrows=limit)
                    for _, row in df.iterrows():
                        # Handle different column naming conventions
                        q_col = 'quizzes' if 'quizzes' in df.columns else 'puzzle'
                        s_col = 'solutions' if 'solutions' in df.columns else 'solution'
                        
                        quiz = np.array([int(c) for c in str(row[q_col])])
                        solution = np.array([int(c) for c in str(row[s_col])])
                        self.data.append((torch.tensor(quiz).long(), torch.tensor(solution).long()))
                    loaded = True
                except Exception as e:
                    print(f"CSV Read failed: {e}")
            else:
                print("\nCRITICAL ERROR: Could not load data.")
                print("1. HuggingFace dataset 'Ritvik19/Sudoku-Dataset' failed.")
                print("2. Local 'sudoku.csv' not found or empty.")
                print("To fix: Download sudoku.csv from Kaggle (https://www.kaggle.com/datasets/bryanpark/sudoku) and place it in this folder.")
                exit()

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def run_sudoku_benchmark():
    print("--- BENCHMARK: 9x9 SUDOKU ---")
    
    # Load Data
    #train_ds = Sudoku1MDataset(split="train", limit=50000)
    
    # Change limit from 50000 to 1000
    train_ds = Sudoku1MDataset(split="train", limit=1000)
    val_ds = Sudoku1MDataset(split="train", limit=100) 
    model = TinyRecursiveModel(
        dim = 256,              
        num_tokens = 10,
        max_seq_len = 81 + 16,  
        network = MLPMixer1D(
            dim = 256,          
            depth = 6,          
            seq_len = 81 
        ),
        num_refinement_blocks = 4, 
        num_latent_refinements = 4 
    )

    trainer = Trainer(
        model,
        train_ds,
        learning_rate = 1e-3,
        weight_decay = 0.0,
        batch_size = 4,        
        accelerate_kwargs = {"gradient_accumulation_steps": 8}, # 4 * 8 = 32 effective batch size
        epochs = 10,
        max_recurrent_steps = 32,
        warmup_steps = 1000,
        compile_model = False,
        cpu = not torch.cuda.is_available()
    )

    trainer.forward()
    
    print("Validating Sudoku...")
    correct_puzzles = 0
    total = len(val_ds)
    
    for i in range(total):
        inp, tgt = val_ds[i]
        inp_gpu = inp.unsqueeze(0).to(trainer.accelerator.device)
        
        pred, _ = model.predict(inp_gpu)
        pred = pred[0]
        
        # Strict accuracy: Entire 81-number grid must match
        if (pred == tgt.to(trainer.accelerator.device)).all():
            correct_puzzles += 1
            
    print(f"Result: {correct_puzzles}/{total} Puzzles Solved Perfectly")

if __name__ == "__main__":
    run_sudoku_benchmark()