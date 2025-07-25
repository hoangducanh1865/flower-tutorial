import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PARTITIONS = 10
BATCH_SIZE = 32