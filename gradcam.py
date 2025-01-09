import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image
from skimage.transform import resize
from torchvision.models import DenseNet121_Weights

IMG_WIDTH, IMG_HEIGHT = 224, 224

# Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.target_layer = self.get_target_layer()
        self.hook()

    def get_target_layer(self):
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                return module
        raise ValueError(f"Layer {self.target_layer_name} not found!")

    def hook(self):
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, _, __, output):
        self.activations = output

    def save_gradients(self, _, __, grad_out):
        self.gradients = grad_out[0]

    def generate_heatmap(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        class_score = output
        self.model.zero_grad()
        class_score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1).squeeze()
        heatmap = torch.clamp(heatmap, min=0).detach().cpu().numpy()
        heatmap = heatmap / np.max(heatmap)  # Normalize heatmap
        return heatmap

    def visualize_heatmap(self, heatmap, image, alpha=0.5):
        heatmap = resize(heatmap, (image.size[1], image.size[0]))
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = plt.cm.jet(heatmap)[:, :, :3]  # Apply colormap
        overlay = np.array(image) * alpha + heatmap * (1 - alpha)
        return overlay.astype(np.uint8)

    def visualize_gradcam(self, image_tensor, heatmap, alpha=0.5):
        # Remove the batch dimension (if it exists) to get a 3D tensor (C, H, W)
        image = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    
        # Normalize the image to [-1, 1]
        image = 2 * ((image - image.min()) / (image.max() - image.min())) - 1  # Normalize to [-1, 1]
    
        # Resize the heatmap to match the image dimensions
        heatmap_resized = resize(heatmap, (image.shape[0], image.shape[1]), mode='reflect')
    
        # Apply the 'jet' colormap to the heatmap and blend it with the original image
        heatmap_overlay = (alpha * plt.cm.jet(heatmap_resized)[:, :, :3] + (1 - alpha) * (image + 1) / 2)  # Adjust for blending
    
        # Clip the values to [0, 1] for proper display
        heatmap_overlay = np.clip(heatmap_overlay, 0, 1)

        return heatmap_overlay

    def generate_image_tensor(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
            DenseNet121_Weights.DEFAULT.transforms(),
        ])

        image = Image.open(image_path).convert('RGB')  # Open the image and convert to RGB

        # Apply the transformations
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Get the model's predictions
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)  # Raw logits
            prediction = (torch.sigmoid(output) > 0.5).int()  # Apply sigmoid and threshold at 0.5
        
        predicted_class = prediction[0, 0]
        
        print(f"Predicted class: {'PNEUMONIA' if predicted_class == 1 else 'NORMAL'}")
        return image_tensor