from collections import OrderedDict
from typing import List, Tuple
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset


# -------------------------------------------------- #
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context
# -------------------------------------------------- #


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
disable_progress_bar()
# print(DEVICE)
# print(flwr.__version__)
# print(torch.__version__)


NUM_PARTITIONS = 10
BATCH_SIZE = 32


def load_datasets(partition_id: int, num_partitions: int):
    fds = FederatedDataset(dataset='cifar10', partitioners={'train': num_partitions})
    partition = fds.load_partition(partition_id)

    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    pytorch_transforms = transforms.Compose (
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        batch['img'] = [pytorch_transforms(img) for img in batch['img']]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test['train'], batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(partition_train_test['test'], batch_size=BATCH_SIZE)
    testset = fds.load_split('test').with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    return trainloader, valloader, testloader


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def train(net, trainloader, epochs: int, verbose=False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()

    for epoch in range(epochs):
        # accuracy = correct / toal
        # epoch_loss = sum(batch_loss) / size_of_train_set
        correct, total, epoch_loss = 0, 0, 0.0

        for batch in trainloader:
            images, labels = batch['img'].to(DEVICE), batch['label'].to(DEVICE) # QUESTION: Why need ".to(DEVICE)"?
            optimizer.zero_grad() # Re-initialize graddients
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward() # Calculate gradients
            optimizer.step() # Update parameters

            epoch_loss += loss
            total += labels.size(0) # QUESTION: ?
            correct += ((torch.max(outputs.data, 1))[1] == labels).sum().item()

        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for batch in testloader:
            images, labels = batch['img'].to(DEVICE), batch['label'].to(DEVICE)
            outputs = net(images)
            loss = loss + criterion(outputs, labels).item()
            predicted = (torch.max(outputs.data, 1))[1] # QUESTION: ?
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()

    loss = loss / len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


# Update the local model with parameters received from the server
def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys() , parameters)
    state_dict = OrderedDict({k : torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# Get the updated model parameters from the local model
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {} # QUESTION 1: Why do we need "len(self.trainloader)"?
                                                                   # QUESTION 2: What is {}?

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    

def client_fn(context: Context) -> Client:
    net = Net().to(DEVICE)

    partition_id = context.node_config['partition-id']
    trainloader, valloader, _ = load_datasets(partition_id=partition_id, num_partitions=NUM_PARTITIONS) # Added num_partitions

    return FlowerClient(net, trainloader, valloader).to_client()


client = ClientApp(client_fn=client_fn)


net = Net()
params = get_parameters(net)


def server_fn(context: Context) -> ServerAppComponents:
  strategy = FedAvg(
      fraction_fit=0.3,
      fraction_evaluate=0.3,
      min_fit_clients=3,
      min_evaluate_clients=3,
      min_available_clients=NUM_PARTITIONS,
      initial_parameters=ndarrays_to_parameters(params)
  )
  config = ServerConfig(num_rounds=3)
  return ServerAppComponents(strategy=strategy, config=config)


server = ServerApp(server_fn=server_fn)


if DEVICE == "cuda":
  backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
  # backend_config = {"client_resources": {"num_gpus": 1}}
else:
  backend_config = {"client_resources": None}

run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_PARTITIONS,
    backend_config=backend_config
)


def server_fn(context: Context) -> ServerAppComponents:
    strategy = FedAdagrad(
        fraction_fit=0.3,
        fraction_evaluate=0.3,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=NUM_PARTITIONS,
        initial_parameters=ndarrays_to_parameters(params),
    )
    # Configure the server for 3 rounds of training
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)


server = ServerApp(server_fn=server_fn)

run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_PARTITIONS,
    backend_config=backend_config,
)


# Function to evaluate the aggregated model
def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]: # QUESTION: ?
  net = Net().to(DEVICE)
  testloader = load_datasets(0, NUM_PARTITIONS)[2]
  set_parameters(net, parameters)
  loss, accuracy = test(net, testloader)
  print(f"Server-side evaluation loss {loss}, accuracy {accuracy}")
  return loss, {"accuracy": accuracy}


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


# Create the ServerApp
server = ServerApp(server_fn=server_fn)


# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_PARTITIONS,
    backend_config=backend_config,
)


class FlowerClient(NumPyClient):
    def __init__(self, pid, net, trainloader, valloader):
        self.pid = pid  # partition ID of a client
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.pid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # Read values from config
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]

        # Use values provided by the config
        print(f"[Client {self.pid}, round {server_round}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=local_epochs)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.pid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(context: Context) -> Client:
    net = Net().to(DEVICE)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, _ = load_datasets(partition_id, num_partitions)
    return FlowerClient(partition_id, net, trainloader, valloader).to_client()


# Create the ClientApp
client = ClientApp(client_fn=client_fn)


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
    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.3,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=NUM_PARTITIONS,
        initial_parameters=ndarrays_to_parameters(params),
        evaluate_fn=evaluate,
        on_fit_config_fn=fit_config,  # Pass the fit_config function
    )
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)


# Create the ServerApp
server = ServerApp(server_fn=server_fn)

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_PARTITIONS,
    backend_config=backend_config,
)


NUM_PARTITIONS = 1000


def fit_config(server_round: int):
    config = {
        "server_round": server_round,
        "local_epochs": 3,
    }
    return config


def server_fn(context: Context) -> ServerAppComponents:
    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=0.025,  # Train on 25 clients (each round)
        fraction_evaluate=0.05,  # Evaluate on 50 clients (each round)
        min_fit_clients=20,
        min_evaluate_clients=40,
        min_available_clients=NUM_PARTITIONS,
        initial_parameters=ndarrays_to_parameters(params),
        on_fit_config_fn=fit_config,
    )
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)


# Create the ServerApp
server = ServerApp(server_fn=server_fn)

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_PARTITIONS,
    backend_config=backend_config,
)


