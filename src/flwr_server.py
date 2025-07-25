from model.model import Net
from utils import get_parameters, set_parameters
from data import load_datasets
from config import DEVICE, NUM_PARTITIONS
from train import train, test
from typing import Tuple, Dict, Optional
from flwr.common import NDArrays, Scalar, Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


net = Net()
params = get_parameters(net)


# Function to evaluate the aggregated model
def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]: # QUESTION: ?
  net = Net().to(DEVICE)
  testloader = load_datasets(0, NUM_PARTITIONS)[2]
  set_parameters(net, parameters)
  loss, accuracy = test(net, testloader)
  print(f"Server-side evaluation loss {loss}, accuracy {accuracy}")
  return loss, {"accuracy": accuracy}


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def server_fn(context: Context) -> ServerAppComponents:
    # Create the FedAvg strategy
    strategy = FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.3,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=NUM_PARTITIONS,
        initial_parameters=ndarrays_to_parameters(params),
        evaluate_fn=evaluate,  # Pass the evaluation function
    )
    # Configure the server for 3 rounds of training
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)


server = ServerApp(server_fn=server_fn)