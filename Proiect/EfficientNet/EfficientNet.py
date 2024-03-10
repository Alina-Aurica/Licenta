import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import EfficientNet_B7_Weights
from tqdm import tqdm
from LoadImagesEfficientNet import *

# aici nu stiu daca e complet
model = models.efficientnet_b7(EfficientNet_B7_Weights.DEFAULT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

# -- TRAINING PART --
def train_function():
    loss = 0
    accuracy = 0
    for epoch in range(num_epochs):
        running_correct = 0
        running_samples = 0
        pbar = tqdm(enumerate(train_loader, 0),
                    unit='image',
                    total=len(train_loader),
                    smoothing=0)

        model.train()
        for i, data in pbar:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            running_samples += labels.size(0)
            accuracy = running_correct / running_samples * 100
            pbar.set_description(
                'Train [ E {}, L {:.4f}, A {:.4f}]'.format(epoch, float(loss) / (i + 1), accuracy))

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Accuracy: {accuracy:.2f}')
        val_function()


def val_function():
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        correct = 0
        total = 0
        for i, data in val_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()
    print(f'Validation Loss: {total_loss / len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')


log_dir = "/tensorboard/test6EfficientNet"
writer = SummaryWriter(log_dir=log_dir)
train_function()
torch.save(model.state_dict(), 'efficientnet_model.pth')




