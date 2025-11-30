import torch
from torch.utils.data import Dataset
from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D
from tiny_recursive_model.trm import TinyRecursiveModel
from tiny_recursive_model.trainer import Trainer

print("--- DIAGNOSTIC SUITE: REASONING & MEMORY ---")

class ParityDataset(Dataset):
    def __init__(self, samples=3000, seq_len=16):
        self.seq_len = seq_len
        self.data = torch.randint(0, 2, (samples, seq_len))
        self.targets = self.data.cumsum(dim=1) % 2 

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]

class AssociativeRecallDataset(Dataset):
    def __init__(self, samples=2000, seq_len=32):
        self.seq_len = seq_len
        self.samples = samples
    
    def __len__(self): return self.samples
    def __getitem__(self, idx):
        keys = torch.randint(0, 10, (self.seq_len // 2,))
        values = keys + 10 
        seq = torch.stack((keys, values), dim=1).flatten()
        query_idx = torch.randint(0, len(keys), (1,))
        query_key = keys[query_idx]
        target_val = values[query_idx]
        inp = torch.cat((seq, query_key))
        tgt = torch.full_like(inp, -100)
        tgt[-1] = target_val.item()
        return inp, tgt

def run_task(name, dataset, epochs=15, num_tokens=256, lr=1e-3):
    print(f"\n[TEST] Starting Task: {name}")
    
    sample_inp, _ = dataset[0]
    actual_seq_len = sample_inp.shape[0]
    
    model = TinyRecursiveModel(
        dim = 64,
        num_tokens = num_tokens,
        max_seq_len = actual_seq_len + 16,
        network = MLPMixer1D(
            dim = 64, 
            depth = 2, 
            seq_len = actual_seq_len 
        ),
        num_refinement_blocks = 2, 
        num_latent_refinements = 4 
    )

    trainer = Trainer(
        model,
        dataset,
        learning_rate = lr,
        weight_decay = 0.0,
        batch_size = 32,
        epochs = epochs,
        max_recurrent_steps = 12,
        warmup_steps = 10,
        compile_model = True,
        cpu = not torch.cuda.is_available()
    )

    trainer.forward()
    
    print(f"      Validating {name}...")
    
    # --- UPGRADE: CHECK BATCH ACCURACY (Luck vs Skill) ---
    # We grab a whole batch (32 puzzles) and check how many it gets right.
    inp_batch, tgt_batch = next(iter(trainer.dataloader))
    inp_batch = inp_batch.to(trainer.accelerator.device)
    tgt_batch = tgt_batch.to(trainer.accelerator.device)
    
    # Run Inference
    preds, _ = model.predict(inp_batch)
    
    # Check Last Token Accuracy
    # (Parity needs to be right at the end; Recall needs to be right at the end)
    correct_count = (preds[:, -1] == tgt_batch[:, -1]).sum().item()
    accuracy = (correct_count / 32) * 100
    
    print(f"      Batch Accuracy: {accuracy:.2f}% ({correct_count}/32)")
    
    if accuracy > 95.0:
        print(f"      >>> SUCCESS: {name} Passed (Intelligence).")
    elif accuracy > 55.0:
        print(f"      >>> INCONCLUSIVE: {name} is learning but not perfect.")
    else:
        print(f"      >>> FAILURE: {name} Failed (Just Guessing).")

if __name__ == "__main__":
    # Task 1: Parity Check
    # LR 2e-3 (Aggressive) to break the 0.65 stagnation
    run_task("Parity Check", ParityDataset(samples=3000, seq_len=16), epochs=20, num_tokens=2, lr=2e-3)
    
    # Task 2: Associative Recall
    run_task("Associative Recall", AssociativeRecallDataset(samples=2000, seq_len=32), epochs=20, num_tokens=256, lr=2e-3)
    