import numpy as np
import torch

from typing import List
from collections import OrderedDict


# Update the local model with parameters received from the server
def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys() , parameters)
    state_dict = OrderedDict({k : torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# Get the updated model parameters from the local model
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]