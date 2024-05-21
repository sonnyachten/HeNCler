import os
import torch
from wcmatch import pathlib
import numpy as np

torch.cuda.empty_cache()
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("CUDA NOT AVAILABLE")
    device = torch.device("cpu")

TensorType = torch.FloatTensor  # HalfTensor, FloatTensor, DoubleTensor
torch.set_default_tensor_type(TensorType)

ROOT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__))) # This is your Project Root
OUT_DIR = pathlib.Path("~/out/hencler/").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = pathlib.Path("~/data").expanduser()
np_rand = np.random.RandomState(42)
