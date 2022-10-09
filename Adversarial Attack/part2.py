import numpy as np
from tqdm import tqdm # Displays a progress bar
import time
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Load the dataset and train, val, test splits
print("Loading datasets...")
FASHION_transform = transforms.Compose([
    transforms.ToTensor(), # Transform from [0,255] uint8 to [0,1] float
])

FASHION_test = datasets.FashionMNIST('.', download=True, train=False, transform=FASHION_transform)
print("Done!")


model_path = './model/fashion_mnist_model.pth'
epsilon = 25/255
batch_size = 1
alpha = 2/255
step_size = [1, 2, 5, 10]

# Create dataloaders
# TODO: Experiment with different batch sizes
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

        #Fully Connected Layer 1
        x = self.fully_connected1(x)
        x = self.relu(x)

        #Fully Connected Layer 2
        x = self.fully_connected2(x)

        return x

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
model = Network().to(device)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

def generate_perturbed_image(img, epsilon, loss_gradient):
    '''
    img - original image
    epsilon - used to adjust the input data by small step
    loss_gradient - gradient of the network
    '''

    perturbed_image = img + epsilon * loss_gradient.sign()
    #to maintain the range of the image in [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image


def evaluate_fsgm(model, loader): # Evalaute on test set
    '''
    This function is used to perform the attack using FAST SIGN GRADIENT METHOD on the Fashion MNIST test set.
    '''

    model.eval() # Set the model to evaluation mode
    adv_examples = []
    correct = 0
    correct_before_perturbation = 0
    incorrect_after_perturbation = 0

    for batch, label in tqdm(loader):
        batch = batch.to(device)
        label = label.to(device)
        batch.requires_grad = True
        pred = model(batch)
        pred_max = torch.argmax(pred,dim=1)
        if pred_max.item() != label.item():
            continue
        correct_before_perturbation += 1
        loss = F.nll_loss(pred, label)
        model.zero_grad()
        loss.backward()
        loss_gradient = batch.grad.data

        perturbed_image = generate_perturbed_image(batch, epsilon, loss_gradient)
        perturbed_image_pred = model(perturbed_image)
        perturbed_image_pred_max = torch.argmax(perturbed_image_pred,dim=1)
        if perturbed_image_pred_max.item() == label.item():
            correct += 1
        else:
            incorrect_after_perturbation += 1
            if len(adv_examples) < 15:
                adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                adv_examples.append( (pred_max.item(), perturbed_image_pred_max.item(), adv_ex) )

    # acc = correct/len(loader.dataset)
    acc = incorrect_after_perturbation / correct_before_perturbation
    print(".......Fast Sign Gradient Method adversarial attack.......")
    print("Number of images predicted correctly before attack: ", correct_before_perturbation)
    print("Number of images predicted incorrectly after attack: ", incorrect_after_perturbation)
    print("Accuracy after attack(incorrectly predicted after attack/corrected predicted before attack): {}".format(acc))

    # Return the accuracy and an adversarial example
    return acc, adv_examples

def visualize_adversial_ex(initial_name, adversarial_examples):
    '''
    This function is used to visualize the perturbed images.
    '''
    class_dict = { 0: 'T-shirt/top', 1: 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat' , 5 : 'Sandal', 6 : 'Shirt', 7 : 'Sneaker' , 8 : 'Bag' , 9 : 'Ankle boot'}
    for j in range(len(adversarial_examples)):
        orig,adv,ex = adversarial_examples[j]
        plt.title("{} -> {}".format(class_dict[orig], class_dict[adv]))
        plt.imshow(ex, cmap="gray")
        name = initial_name + time.strftime("%Y%m%d-%H%M%S") + '.png'  
        plt.savefig('./' + name)
        plt.show()


def evaluate_pgd(loader, alpha, step_size):
    '''
    This function is used to perform adversarial attack using the non-targeted white box PGD evasion attack
    on the Fashion MNIST test set.
    '''

    model.eval() # Set the model to evaluation mode
    correct = 0
    adv_examples = []
    correct_before_perturbation = 0
    incorrect_after_perturbation = 0

    for batch, label in tqdm(loader):
        batch = batch.to(device)
        original_image = batch
        label = label.to(device)
        pred = model(original_image)
        pred_max = torch.argmax(pred,dim=1)
        if pred_max.item() != label.item():
            continue
        correct_before_perturbation += 1
        for i in range(0, step_size):
            batch.requires_grad = True
            pred = model(batch)
            loss = F.nll_loss(pred, label)
            model.zero_grad()
            loss.backward()
            loss_gradient = batch.grad.data
            perturbed_image = batch + alpha * loss_gradient.sign()
            total_perturbation = perturbed_image - original_image
            perturbed_image = torch.clamp(total_perturbation, -epsilon, epsilon)
            batch.data = perturbed_image + original_image

        perturbed_image_pred = model(batch)
        perturbed_image_pred_max = torch.argmax(perturbed_image_pred,dim=1)
        if perturbed_image_pred_max.item() == label.item():
            correct += 1
        else:
            incorrect_after_perturbation += 1
            if len(adv_examples) < 15:
                adv_ex = batch.squeeze().detach().cpu().numpy()
                adv_examples.append( (pred_max.item(), perturbed_image_pred_max.item(), adv_ex) )

    # acc = correct/len(loader.dataset)
    acc = incorrect_after_perturbation / correct_before_perturbation
    print(".......Non-targeted white-box PGD evasion attack.......")
    print("Steps: ", step_size)
    print("Number of images predicted correctly before attack: ", correct_before_perturbation)
    print("Number of images predicted incorrectly after attack: ", incorrect_after_perturbation)
    print("Accuracy after attack(incorrectly predicted after attack/corrected predicted before attack): {}".format(acc))

    # Return the accuracy and an adversarial example
    return acc, adv_examples

#run the fsgm attack
test_accuracy, adversarial_examples = evaluate_fsgm(model, testloader)
visualize_adversial_ex('fsgm_', adversarial_examples)
#run the pgd attack
for i in step_size:
    accuracy, adversarial_examples = evaluate_pgd(testloader, alpha, i)
    if i == 10:
        visualize_adversial_ex('pgd_',adversarial_examples)



