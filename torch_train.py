import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor

IMAGE_SIZE = 28
HIDDEN_LAYER_SIZE = 512
NO_LABELS = 10

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 5

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(IMAGE_SIZE*IMAGE_SIZE, HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimiser):
    size = len(dataloader.dataset)
    model.train() # train mode
    for batch, (inputs, expected_output) in enumerate(dataloader):
        prediction = model(inputs.to(device))
        loss = loss_fn(prediction, expected_output.to(device))
        
        #backpropogation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad() # reset gradients

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(inputs)
            print(f"loss: {loss:.8f} |{current}/{size}]")

def test_loop(dataloader, model, loss_fn) -> float:
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad(): # disable differentiation shit
        for inputs, expected_output in dataloader:
            prediction = model(inputs.to(device))
            test_loss += loss_fn(prediction, expected_output.to(device)).item()
            correct += (prediction.argmax(1) == expected_output.argmax(1).to(device)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):.8f}%, Avg loss: {test_loss:.8f} \n")
    return correct


if __name__ == "__main__":
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(), # turns image data into a tensor
        target_transform=Lambda(lambda y: torch.zeros(NO_LABELS, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
        # turns label data into one-hot encoded vectors corresponding to the specific label
    )

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

    testing_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(lambda y: torch.zeros(NO_LABELS, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    )

    test_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=True)

    model = NeuralNetwork().to(device)
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    accuracy = 0.0
    epoch_i = 1

    while accuracy < 0.95:
        print(f"EPOCH {epoch_i}")
        train_loop(train_dataloader, model, loss_fn, optimiser)
        accuracy = test_loop(test_dataloader, model, loss_fn)
        epoch_i += 1
    torch.save(model.state_dict(), 'weights/model.dat')

