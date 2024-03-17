import torch
import torchmetrics
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
# aici sa vad de ce optimizer pot alege
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50


# -- TRAINING PART --
def train_function():
    loss = 0
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=25).to(device)
    for epoch in range(num_epochs):
        # running_correct = 0
        # running_samples = 0
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
            # running_correct += (predicted == labels).sum().item()
            # running_samples += labels.size(0)
            # accuracy = running_correct / running_samples * 100
            accuracy_metric.update(predicted, labels)
            pbar.set_description(
                'Train [ E {}, L {:.4f}, A {:.4f}]'.format(epoch, float(loss) / (i + 1), accuracy_metric))

        accuracy = accuracy_metric.compute()
        accuracy_metric.reset()

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Accuracy: {accuracy:.2f}')
        val_function()


def val_function():
    model.eval()

    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=25).to(device)
    precision_metric = torchmetrics.Precision(task="multiclass", num_classes=25, average='macro').to(device)
    recall_metric = torchmetrics.Recall(task="multiclass", num_classes=25, average='macro').to(device)
    f1_score_metric = torchmetrics.F1Score(task="multiclass", num_classes=25, average='macro').to(device)
    confusion_matrix_metric = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=25).to(device)

    with torch.no_grad():
        total_loss = 0.0
        # correct = 0
        # total = 0
        pbar = tqdm(enumerate(val_loader, 0),
                    unit='image',
                    total=len(val_loader),
                    smoothing=0)
        for i, data in pbar:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            accuracy_metric.update(predicted, labels)
            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
            f1_score_metric.update(predicted, labels)
            confusion_matrix_metric.update(predicted, labels)

            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

    total_loss = total_loss / len(val_loader)
    accuracy = accuracy_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1_score = f1_score_metric.compute()
    confusion_matrix = confusion_matrix_metric.compute()

    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_score_metric.reset()
    confusion_matrix_metric.reset()

    model.train()
    print(f'Validation Loss: {total_loss:.4f}, '
          f'Accuracy: {accuracy:.2f}, '
          f'Precision: {precision:.2f}, '
          f'Recall: {recall:.2f}, '
          f'F1 Score: {f1_score:.2f}, '
          f'Confusion Matrix: {confusion_matrix:.2f}')


log_dir = "/tensorboard/test6EfficientNet"
writer = SummaryWriter(log_dir=log_dir)
train_function()
torch.save(model.state_dict(), 'efficientnet_model.pth')
