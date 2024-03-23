# import torch
# import torchvision
import torchmetrics
import torchvision.models as models

from LoadImagesEfficientNet import *
from EfficientNet import *

model = models.efficientnet_b7(pretrained=False)
model_path = 'efficientnet_model_3LR_Adam.pth'
model.load_state_dict(torch.load(model_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def test_function():
    model.eval()

    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=25).to(device)
    precision_metric = torchmetrics.Precision(task="multiclass", num_classes=25, average='macro').to(device)
    recall_metric = torchmetrics.Recall(task="multiclass", num_classes=25, average='macro').to(device)
    f1_score_metric = torchmetrics.F1Score(task="multiclass", num_classes=25, average='macro').to(device)

    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth:', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    pbar = tqdm(enumerate(test_loader, 0),
                unit='image',
                total=len(test_loader),
                smoothing=0)

    # for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    # Testing loop
    with torch.no_grad():
        for i, data in pbar:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Update metrics
            accuracy_metric.update(predicted, labels)
            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
            f1_score_metric.update(predicted, labels)

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    accuracy = accuracy_metric.compute() * 100
    precision = precision_metric.compute() * 100
    recall = recall_metric.compute() * 100
    f1_score = f1_score_metric.compute() * 100

    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_score_metric.reset()

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1_score:.2f}')

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


test_function()
