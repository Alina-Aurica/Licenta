# import torch
# import torchvision
import torchvision.models as models

from LoadImagesEfficientNet import *
from EfficientNet import *

# iterator from the test Data Loader
dataiter = iter(test_loader)
images, labels = next(dataiter)

# print images - as a grid
imshow(torchvision.utils.make_grid(images))
# print the Ground Truth labels (the true labels) for images displayed
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(32)))

# re-load the model
model = models.efficientnet_b7(EfficientNet_B7_Weights.DEFAULT)
model.load_state_dict(torch.load(model.state_dict(), 'efficientnet_model.pth')) # cred ca e okay asa

# computes the outputes of images (which images?)
outputs = model(images)
# select the class with the highest output value - model's confidence
_, predicted = torch.max(outputs, 1)
# print the first 3 images in the batch
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

# overall test accuracy - on test dataset
# init 2 variables
correct = 0
total = 0
# torch.no_grad() - since we're not training
#                   we don't need to calculate
#                   the gradients for our outputs
with torch.no_grad():
    # iterates over test Data Loader
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # if predicted is equal to true labels then it's a TP
        correct += (predicted == labels).sum().item()

# print the accuracy of the network
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# class-wise accuracy
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# PRECISION, RECALL AND ERROR RATES