import os  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torchvision.transforms as transforms  
import torchvision.models as models  
import timm  
from PIL import Image  
from torch.utils.data import Dataset, DataLoader  
from efficientnet_pytorch import EfficientNet  
import cv2  
import numpy as np  
from torchvision.transforms import Compose, Resize, ToTensor, Normalize  

# Define the DenseNet model for depth estimation  
class DenseNetDepthEstimation(nn.Module):  
    def __init__(self):  
        super(DenseNetDepthEstimation, self).__init__()  
        self.densenet = models.densenet121(pretrained=True)  
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, 1)  # Output 1 for depth estimation  

    def forward(self, x):  
        x = self.densenet(x)  
        return x  

# Define the Deep3DBox model for depth estimation  
class Deep3DBoxDepthEstimation(nn.Module):  
    def __init__(self):  
        super(Deep3DBoxDepthEstimation, self).__init__()  
        self.backbone = models.resnet50(pretrained=True)  
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 512)  

        self.depth_regressor = nn.Sequential(  
            nn.Linear(512, 256),  
            nn.ReLU(),  
            nn.Linear(256, 1)  # Output a single value for depth  
        )  

    def forward(self, x):  
        features = self.backbone(x)  
        depth = self.depth_regressor(features)  
        return depth   

# Define the EfficientNet model for depth estimation  
class EfficientNetDepthEstimation(nn.Module):  
    def __init__(self, model_name='efficientnet-b0'):  
        super(EfficientNetDepthEstimation, self).__init__()  
        self.efficientnet = EfficientNet.from_pretrained(model_name)  
        self.efficientnet._fc = nn.Linear(self.efficientnet._fc.in_features, 1)  # Output 1 for depth estimation  

    def forward(self, x):  
        return self.efficientnet(x)  

# Define the RegNet model for depth estimation  
class RegNetDepthEstimation(nn.Module):  
    def __init__(self, model_name='regnety_400mf'):  
        super(RegNetDepthEstimation, self).__init__()  
        # Load the RegNet model  
        self.regnet = timm.create_model(model_name, pretrained=True)  
        # Modify the classifier to output a single value for depth estimation  
        self.regnet.fc = nn.Linear(self.regnet.fc.in_features, 1)  

    def forward(self, x):  
        return self.regnet(x)  

# Custom dataset class to load images for depth estimation  
class DepthEstimationDataset(Dataset):  
    def __init__(self, image_dir, transform=None):  
        self.image_dir = image_dir  
        self.transform = transform  
        self.image_filenames = os.listdir(image_dir)  

    def __len__(self):  
        return len(self.image_filenames)  

    def __getitem__(self, idx):  
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])  
        image = Image.open(img_name).convert('RGB')  

        if self.transform:  
            image = self.transform(image)  

        return image  

# Define image transformations  
transform = transforms.Compose([  
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])  

# Prepare dataset and dataloaders  
image_dir = './processed_depth_maps'  # Directory with processed images  
train_dataset = DepthEstimationDataset(image_dir=image_dir, transform=transform)  
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
val_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # Placeholder for validation data, replace when available  

# Instantiate models and setup training configurations  
model1 = DenseNetDepthEstimation()  
model2 = Deep3DBoxDepthEstimation()  
model3 = EfficientNetDepthEstimation()  
model4 = RegNetDepthEstimation()  # New RegNet model instantiation  

# Define the loss function and optimizer for each model  
criterion = nn.L1Loss()  # You can also use nn.MSELoss() if preferred  
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)  
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)  
optimizer3 = optim.Adam(model3.parameters(), lr=0.001)  
optimizer4 = optim.Adam(model4.parameters(), lr=0.001)  # Optimizer for RegNet model  

# Training loop  
num_epochs = 10  # Specify the number of epochs  

def train_model(model, optimizer, train_loader, epoch):  
    model.train()  
    for inputs in train_loader:  
        optimizer.zero_grad()  
        
        # Generate dummy targets (replace with actual targets)  
        targets = torch.rand((inputs.shape[0], 1))  # Dummy targets  

        outputs = model(inputs)  
        loss = criterion(outputs, targets)  
        loss.backward()  
        optimizer.step()  

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')  

for epoch in range(num_epochs):  
    train_model(model1, optimizer1, train_loader, epoch)  
    train_model(model2, optimizer2, train_loader, epoch)  
    train_model(model3, optimizer3, train_loader, epoch)  
    train_model(model4, optimizer4, train_loader, epoch)  # Training RegNet model  

# Evaluation loop  
def evaluate_model(model, val_loader):  
    model.eval()  
    total_loss = 0  
    with torch.no_grad():  
        for inputs in val_loader:  
            outputs = model(inputs)  
            targets = torch.rand((outputs.size(0), 1))  # Dummy targets  
            loss = criterion(outputs, targets)  
            total_loss += loss.item()  

    avg_loss = total_loss / len(val_loader)  
    print(f'Validation Loss: {avg_loss:.4f}')  

# Evaluate all models  
evaluate_model(model1, val_loader)  
evaluate_model(model2, val_loader)  
evaluate_model(model3, val_loader)  
evaluate_model(model4, val_loader)  # Evaluate RegNet model  

# Function to estimate depth maps using MiDaS model  
def estimate_depth_maps(image_paths, output_directory, model_type="DPT_Large"):  
    # Load the MiDaS model  
    model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)  
    model.eval()  # Set the model to evaluation mode  

    # Check if output directory exists, if not, create it  
    if not os.path.exists(output_directory):  
        os.makedirs(output_directory)  

    # Define the transformations  
    transform = Compose([  
        Resize(384),  # Resize to the model's input size  
        ToTensor(),  
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])  

    for image_path in image_paths:  
        # Load and preprocess the image  
        image = Image.open(image_path).convert("RGB")  
        input_tensor = transform(image).unsqueeze(0)  

        # Perform depth estimation  
        with torch.no_grad():  
            depth_map = model(input_tensor)  

        # Post-process the depth map  
        depth_map = depth_map.squeeze().cpu().numpy()  
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # Normalize  
        depth_map = (depth_map * 255).astype(np.uint8)  

        # Create a unique filename for the depth map output  
        base_name = os.path.basename(image_path)  
        depth_map_filename = os.path.join(output_directory, f'depth_map_{base_name}')  
        
        # Save the depth map  
        cv2.imwrite(depth_map_filename, depth_map)  
        print(f"Depth map saved as {depth_map_filename} successfully.")  

# Example usage for MiDaS model  
color_mapped_images = [f'./output_colors/color_mapped_image_{i}.jpg' for i in range(1, 2001)]  # Replace with your actual paths  
output_dir = './depth_maps'  # Directory to save the depth maps  

# Call the function to estimate depth maps  
estimate_depth_maps(color_mapped_images, output_dir, model_type="DPT_Large")
