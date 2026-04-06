import struct
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from array import array
from torchvision.transforms import Lambda, ToTensor
from os.path import join
import numpy as np

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
            print(f"training loss: {loss:.8f} [{current}/{size}]")

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
    print(f"Test Error: \n Accuracy: {(100*correct):.8f}%, Avg loss: {test_loss:.8f}")
    return correct

clamper = lambda x: 1.0 if x > 128 else 0.0
clamper_func = np.vectorize(clamper)

# parts taken from https://www.kaggle.com/code/hojjatk/read-mnist-dataset,
class CustomMNIST(Dataset):
    def __init__(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        encoded_labels = np.zeros((len(labels), 10), dtype=float)
        encoded_labels[np.arange(len(labels)), labels] = 1
        labels = encoded_labels
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        

        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = np.array(clamper_func(img))
            img = img.reshape(28, 28)
            images[i][:] = img       
        self.x = torch.from_numpy(np.array(images)).to(dtype=torch.float)
        self.y = torch.from_numpy(np.array(labels)).to(dtype=torch.float)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

if __name__ == "__main__":
    # data per batch format, where n is the batch size
    # expected_output: n one-hot encoded vectors of size 10, corresponding to label
        # size is n, 10
    # inputs: tensor of n elements, each containing 1 tensor, in each tensor is a 28x28 matrix.
        # size is n, 1, 28, 28, all values between 0 and 1

    input_path = 'data/MNIST/raw/'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte')

    training_data = CustomMNIST(training_images_filepath, training_labels_filepath)
    testing_data = CustomMNIST(test_images_filepath, test_labels_filepath)
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=True)

    model = NeuralNetwork().to(device)
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    accuracy = 0.0
    epoch_i = 1

    while accuracy < 0.95 and epoch_i <= 100:
        print(f"EPOCH {epoch_i}")
        train_loop(train_dataloader, model, loss_fn, optimiser)
        accuracy = test_loop(test_dataloader, model, loss_fn)
        epoch_i += 1
    torch.save(model.state_dict(), 'weights/custom_model.dat')

