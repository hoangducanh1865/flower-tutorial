import torch
from src.config.config import DEVICE

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