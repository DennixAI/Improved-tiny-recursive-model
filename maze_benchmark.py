import torch
import numpy as np
from torch.utils.data import Dataset
from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D
from tiny_recursive_model.trm import TinyRecursiveModel
from tiny_recursive_model.trainer import Trainer

# MAZE GENERATOR (Recursive Backtracker) 
def generate_maze(size=9):
    # 0=Empty, 1=Wall, 2=Start, 3=Goal, 4=Path
    maze = np.ones((size, size), dtype=int)
    
    def carve(x, y):
        maze[y, x] = 0
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        np.random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and maze[ny, nx] == 1:
                maze[y + dy//2, x + dx//2] = 0
                carve(nx, ny)
    
    carve(1, 1)
    
    # Set Start (2) and Goal (3)
    maze[1, 1] = 2
    maze[size-2, size-2] = 3
    
    # Solve to find Target Path (4)
    q = [(1, 1, [])]
    visited = set()
    correct_path_grid = maze.copy()
    
    solved = False
    while q:
        x, y, path = q.pop(0)
        if (x, y) == (size-2, size-2):
            for px, py in path:
                if correct_path_grid[py, px] == 0:
                    correct_path_grid[py, px] = 4 # Mark path
            solved = True
            break
        
        if (x, y) in visited: continue
        visited.add((x, y))
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and maze[ny, nx] != 1:
                q.append((nx, ny, path + [(nx, ny)]))
                
    if not solved: return generate_maze(size)
    
    return maze.flatten(), correct_path_grid.flatten()

class MazeDataset(Dataset):
    def __init__(self, samples=1000, size=9):
        self.data = []
        for _ in range(samples):
            inp, tgt = generate_maze(size)
            self.data.append((torch.tensor(inp).long(), torch.tensor(tgt).long()))
            
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


def run_maze_test():
    print("\n[TEST] Starting Task: 2D Maze Solving (9x9)")
    
    MAZE_SIZE = 9
    SEQ_LEN = MAZE_SIZE * MAZE_SIZE
    NUM_TOKENS = 5 
    
    print("      Generating 5,000 Mazes...")
    dataset = MazeDataset(samples=5000, size=MAZE_SIZE)
    
    model = TinyRecursiveModel(
        dim = 128,              
        num_tokens = NUM_TOKENS,
        max_seq_len = SEQ_LEN + 16,
        network = MLPMixer1D(
            dim = 128, 
            depth = 4,
            seq_len = SEQ_LEN 
        ),
        num_refinement_blocks = 2, 
        num_latent_refinements = 4 
    )

    trainer = Trainer(
        model,
        dataset,
        learning_rate = 1e-4,   # Low LR for precision
        weight_decay = 0.0,     # Critical
        batch_size = 8,
        accelerate_kwargs = {
            "mixed_precision": "fp16",
            "gradient_accumulation_steps": 4
        },
        epochs = 50,            # 50 Epochs for SOTA results
        max_recurrent_steps = 32, 
        warmup_steps = 1000,
        compile_model = False,
        cpu = not torch.cuda.is_available(),
        checkpoint_folder = './maze_checkpoints'

    )

    print("      Training (Auto-Resuming if checkpoints exist)...")
    trainer.forward()
    
    print("      Validating Maze Solver...")
    inp, tgt = dataset[0]
    inp_gpu = inp.unsqueeze(0).to(trainer.accelerator.device)
    
    pred, _ = model.predict(inp_gpu)
    pred_cpu = pred[0].cpu().numpy().reshape(MAZE_SIZE, MAZE_SIZE)
    tgt_cpu = tgt.numpy().reshape(MAZE_SIZE, MAZE_SIZE)
    
    # Visualization
    print("\n--- Target Path ---")
    print(tgt_cpu)
    print("\n--- Predicted Path ---")
    print(pred_cpu)
    
    # Metrics
    mask = (tgt_cpu == 4)
    correct_steps = (pred_cpu[mask] == 4).sum()
    total_steps = mask.sum()
    
    print(f"\nPath Accuracy: {correct_steps}/{total_steps} steps found")
    
    if correct_steps == total_steps:
        print(">>> SUCCESS: Maze Solved Perfectly.")
    elif correct_steps > 0:
        print(">>> PARTIAL: Found part of the path.")
    else:
        print(">>> FAILURE: Lost in the maze.")

if __name__ == "__main__":
    run_maze_test()