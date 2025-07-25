from flwr_datasets import FederatedDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from config import BATCH_SIZE


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