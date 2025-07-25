# from src.flwr_client import client
# from src.flwr_server import server
# from src.config import NUM_PARTITIONS, DEVICE
# from flwr.simulation import run_simulation

# backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}} if DEVICE == "cuda" else {"client_resources": None}

# run_simulation(
#     server_app=server,
#     client_app=client,
#     num_supernodes=NUM_PARTITIONS,
#     backend_config=backend_config,
# )


# main.py
# WARNING: This only works for `flwr.simulation.start_simulation` (not actual distributed)
from src.flwr_server import start_server
from src.flwr_client import client_fn

import flwr as fl
from flwr.simulation import start_simulation

if __name__ == "__main__":
    start_simulation(
        client_fn=client_fn,
        num_clients=10,
        config=fl.server.ServerConfig(num_rounds=3),
    )