import pickle
from config import *

# your final sample count from log
N = 1119100

with open(SHAPES_V2_PATH, "wb") as f:
    pickle.dump({"n_samples": N}, f)

print("array_shapes_v2.pkl created")