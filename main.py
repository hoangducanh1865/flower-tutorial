from src.flwr_client import client
from src.flwr_server import server
from src.config.config import NUM_PARTITIONS, DEVICE
from flwr.simulation import run_simulation

# import sys
# import os


# sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}} if DEVICE == "cuda" else {"client_resources": None}

run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_PARTITIONS,
    backend_config=backend_config,
)
