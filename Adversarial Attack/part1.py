import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt

# Load the dataset and train, val, test splits
print("Loading datasets...")
FASHION_transform = transforms.Compose([
    transforms.ToTensor(), # Transform from [0,255] uint8 to [0,1] float
])
FASHION_trainval = datasets.FashionMNIST('.', download=True, train=True, transform=FASHION_transform)
FASHION_train = Subset(FASHION_trainval, range(50000))
FASHION_val = Subset(FASHION_trainval, range(50000,60000))
FASHION_test = datasets.FashionMNIST('.', download=True, train=False, transform=FASHION_transform)
print("Done!")

#create model directory to save the trained model
if not os.path.exists('./model'):
    os.makedirs('./model')

#hyperparameters for training the model
batch_size = 32
learning_rate = 0.001
weight_decay = 1e-4
epoch = 40
model_path = './model/fashion_mnist_model.pth'

# Create dataloaders
# TODO: Experiment with different batch sizes
trainloader = DataLoader(FASHION_train, batch_size, shuffle=True)
valloader = DataLoader(FASHION_val, batch_size, shuffle=True)
testloader = DataLoader(FASHION_test, batch_size, shuffle=True)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Design your own network, define layers here.
        # Here We provide a sample of two-layer fully-connected network from HW4 Part3.
        # Your solution, however, should contain convolutional layers.
        # Refer to PyTorch documentations of torch.nn to pick your layers. (https://pytorch.org/docs/stable/nn.html)
        # Some common Choices are: Linear, Conv2d, ReLU, MaxPool2d, AvgPool2d, Dropout
        # If you have many layers, consider using nn.Sequential() to simplify your code
        # self.fc1 = nn.Linear(28*28, 8) # from 28x28 input image to hidden layer of size 256
        # self.fc2 = nn.Linear(8,10) # from hidden layer to 10 class scores

        #defining the layers for lenet network: three convlutional layer, 2 fully connected layer, pooling layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels= 120, kernel_size=(4,4), stride=(1,1), padding=(0,0))

        self.fully_connected1 = nn.Linear(in_features=120, out_features=84)
        self.fully_connected2=nn.Linear(in_features=84, out_features=10)

        self.pooling_layer = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        # # TODO: Design your own network, implement forward pass here
        # x = x.view(-1,28*28) # Flatten each image in the batch
        # x = self.fc1(x)
        # relu = nn.ReLU() # No need to define self.relu because it contains no parameters
        # x = relu(x)
        # x = self.fc2(x)
        # # The loss layer will be applied outside Network class

        #Convolution Layer 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling_layer(x)
        
        #Convolution Layer 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling_layer(x)
        x = self.dropout(x)

        #Convolution Layer 3
        x = self.conv3(x)
        x = self.relu(x)

        #flatten x
        x = x.view(-1, 120)

        #Fully connected layer 1
        x = self.fully_connected1(x)
        x = self.relu(x)

        #Fully connected layer 2
        x = self.fully_connected2(x)

        return x

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
model = Network().to(device)
# TODO: Define loss function 
criterion = nn.CrossEntropyLoss() # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength

#training and validation loss after every epoch
training_loss = []
validation_loss = []
def train(model, loader, valloader, num_epoch = 10): # Train the model
    print("Start training...")
    for i in tqdm(range(num_epoch)):
        model.train() # Set the model to training mode
        train_running_loss = []
        for batch, label in loader:
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            train_running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights
        print("Epoch {} training loss:{}".format(i+1,np.mean(train_running_loss))) # Print the average loss for this epoch
        training_loss.append(np.mean(train_running_loss))

        #run the model to evaluate on the validation dataset and get the validation accuracy
        model.eval()
        val_running_loss = []
        for val_batch, val_label in valloader:
            val_batch = val_batch.to(device)
            val_label = val_label.to(device)
            pred = model(val_batch) # This will call Network.forward() that you implement
            loss_val = criterion(pred, val_label) # Calculate the loss
            val_running_loss.append(loss_val.item())
        print("Epoch {} validation loss:{}".format(i+1,np.mean(val_running_loss))) # Print the average loss for this epoch
        validation_loss.append(np.mean(val_running_loss))
        print("Evaluation on training data...")
        evaluate(model, trainloader)
    #save the model once all the epochs are complete
    torch.save(model.state_dict(), model_path)
    print("Done!")

def plot_loss():
    x = np.arange(0, len(training_loss), 1, dtype=int)
    plt.plot(x, training_loss, label = 'Training Loss')
    plt.plot(x, validation_loss, label = 'Validation Loss')
    plt.title("Training and Validation Plots")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./loss_plot.png')
    plt.show()


def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    correct = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            correct += (torch.argmax(pred, dim=1) == label).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    return acc

#run training for epoch number of times
train(model, trainloader, valloader, epoch)
#visualize the training and validation loss
plot_loss()
print("Evaluate on validation set...")
evaluate(model, valloader)
print("Evaluate on test set")
evaluate(model, testloader)
