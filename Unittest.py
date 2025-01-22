import unittest  
import torch  
from torchvision.transforms import Compose, Resize, ToTensor, Normalize  
from my_depth_estimation_module import (  
    DenseNetDepthEstimation,  
    Deep3DBoxDepthEstimation,  
    EfficientNetDepthEstimation,  
    RegNetDepthEstimation,  
    DepthEstimationDataset,  
    train_model,  
    evaluate_model  
)  

class TestDepthEstimationModels(unittest.TestCase):  
    
    def setUp(self):  
        # This method will run before each test  
        self.input_tensor = torch.rand((1, 3, 224, 224))  # A random tensor simulating a batch of images  
        self.model1 = DenseNetDepthEstimation()  
        self.model2 = Deep3DBoxDepthEstimation()  
        self.model3 = EfficientNetDepthEstimation()  
        self.model4 = RegNetDepthEstimation()  

    def test_densenet_forward(self):  
        output = self.model1(self.input_tensor)  
        self.assertEqual(output.shape, (1, 1), "DenseNet output should be of shape (1, 1)")  

    def test_deep3d_box_forward(self):  
        output = self.model2(self.input_tensor)  
        self.assertEqual(output.shape, (1, 1), "Deep3DBox output should be of shape (1, 1)")  

    def test_efficientnet_forward(self):  
        output = self.model3(self.input_tensor)  
        self.assertEqual(output.shape, (1, 1), "EfficientNet output should be of shape (1, 1)")  

    def test_regnet_forward(self):  
        output = self.model4(self.input_tensor)  
        self.assertEqual(output.shape, (1, 1), "RegNet output should be of shape (1, 1)")  

class TestDepthEstimationDataset(unittest.TestCase):  

    def setUp(self):  
        # Create a temporary directory and some dummy images for testing  
        self.image_dir = './test_images'  # Ensure you create this directory and add images  
        self.transform = Compose([  
            Resize((224, 224)),  
            ToTensor(),  
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])  
        self.dataset = DepthEstimationDataset(image_dir=self.image_dir, transform=self.transform)  

    def test_dataset_length(self):  
        self.assertGreater(len(self.dataset), 0, "Dataset should have images")  

    def test_get_item(self):  
        sample = self.dataset[0]  
        self.assertEqual(sample.shape, (3, 224, 224), "Sample shape should be (3, 224, 224)")  

class TestTrainingEvaluation(unittest.TestCase):  
    
    def setUp(self):  
        # Model and optimizer setup  
        self.model = DenseNetDepthEstimation()  
        self.model.train()  # Set to training mode  
        self.dummy_loader = torch.utils.data.DataLoader([torch.rand((3, 224, 224))] * 10)  # Dummy data  

    def test_training_step(self):  
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  
        loss_before = torch.nn.L1Loss()(self.model(torch.rand(1, 3, 224, 224)), torch.rand(1, 1))  
        train_model(self.model, optimizer, self.dummy_loader, 0)  # Call your train function with dummy data  
        loss_after = torch.nn.L1Loss()(self.model(torch.rand(1, 3, 224, 224)), torch.rand(1, 1))  
        self.assertLess(loss_after.item(), loss_before.item(), "Loss should decrease after training step.")  

    def test_evaluation_step(self):  
        loss = evaluate_model(self.model, self.dummy_loader)  
        self.assertIsInstance(loss, float, "Evaluation loss should be a float.")  

if __name__ == '__main__':  
    unittest.main()
    