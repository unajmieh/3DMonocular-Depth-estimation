import os  
import torch  
from PIL import Image  
from torch.utils.data import Dataset  
import torchvision.transforms as transforms  
from torch.utils.data import DataLoader  

# Define transformations for the images  
transform = transforms.Compose([  
    transforms.Resize((224, 224)),  # Resize images to 224x224  
    transforms.ToTensor(),  # Convert images to PyTorch tensors  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats  
])

class CamVidDataset(Dataset):  
    def __init__(self, image_dir, transform=None):  
        self.image_dir = image_dir  
        self.transform = transform  
        # List of image files in the directory  
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]  # Adjust extensions as needed  

    def __len__(self):  
        return len(self.image_filenames)  

    def __getitem__(self, idx):  
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])  
        image = Image.open(img_name).convert('RGB')  # Load image and ensure it's in RGB format  

        if self.transform:  
            image = self.transform(image)  

        return image  # Return the preprocessed image

image_dir = './Dataset/'  # Replace with the actual path to your images  
camvid_dataset = CamVidDataset(image_dir=image_dir, transform=transform)  
data_loader = DataLoader(camvid_dataset, batch_size=1, shuffle=False)  # Set batch_size according to your needs   
for batch in data_loader:  
    # batch is a tensor with shape (batch_size, 3, 224, 224) after transformations  
    print(batch.shape)  # Print shape of the batch to verify