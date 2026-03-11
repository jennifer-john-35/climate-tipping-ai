import sys
import os
from utils.grid_generator import generate_global_grid
from utils.data_loader import load_grid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

grid = load_grid()

print("Grid size:", len(grid))
print(grid.head())
