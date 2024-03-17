import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from SplitDatasets import *

# --- COSTUM IMAGES ---
class CustomImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.data)
        (img_path, class_id) = self.data[idx]
        class_id_number = int(class_id)
        image = Image.open(images_folder_path + img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, class_id_number


transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- DATASETS ---
train_data = return_tuple_of_dataset("D:/Facultate/ANUL 4/Licenta/Proiect/splits/train", '00')
val_data = return_tuple_of_dataset("D:/Facultate/ANUL 4/Licenta/Proiect/splits/validation", 0)
test_data = return_tuple_of_dataset("D:/Facultate/ANUL 4/Licenta/Proiect/splits/test", 0)
overfit_data = return_tuple_of_dataset("D:/Facultate/ANUL 4/Licenta/Proiect/splits/overfit", 0)

train_dataset = CustomImageDataset(train_data, transform=transform)
val_dataset = CustomImageDataset(val_data, transform=transform)
test_dataset = CustomImageDataset(test_data, transform=transform)
overfit_dataset = CustomImageDataset(overfit_data, transform=transform)

batch_size = 16 # modific valorile in functie de rezultate

# --- DATALOADERS ---
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# train_loader = DataLoader(overfit_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# val_loader = DataLoader(overfit_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- CLASSES ---
classes = ('BEANS', 'CAKE', 'CANDY', 'CEREAL', 'CHIPS',
           'CHOCOLATE', 'COFFEE', 'CORN', 'FISH', 'FLOUR',
           'HONEY', 'JAM', 'JUICE', 'MILK', 'NUTS',
           'OIL', 'PASTA', 'RICE', 'SODA', 'SPICES',
           'SUGAR', 'TEA', 'TOMATO_SAUCE', 'VINEGAR', 'WATER'
           )

# function to show an image
def imshow(img):
    # convert from PyTorch to numpy array
    img = img.numpy()
    # convert from CxHxW to HxWxC
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
# print labels
# convert the numerical labels to their textual representation
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))