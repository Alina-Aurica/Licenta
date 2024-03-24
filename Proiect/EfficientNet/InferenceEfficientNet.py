import torch
import torchvision.models as models
from PIL import Image
from LoadImagesEfficientNet import transform, classes

model = models.efficientnet_b7(pretrained=False)
model.load_state_dict(torch.load('D:/Facultate/ANUL 4/Licenta/Modele/efficientnet_model_3LR_Adam_aug_v1.pth'))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.to(device)

def predict(image_path):
    image = process_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

image_path = 'D:/Facultate/ANUL 4/Licenta/Licenta/Proiect/inferenceImg/coca-cola.jpg'
predicted_class_index = predict(image_path)

print(f'Predicted class: {classes[predicted_class_index]}')
