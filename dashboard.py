import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from gradcam import GradCAM
from torchvision.models import densenet121
from torchvision.transforms.functional import to_pil_image
from zennit.attribution import Gradient
from zennit.composites import EpsilonGammaBox
from zennit.types import BatchNorm
from zennit.image import imgify
from zennit.rules import Epsilon
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

IMG_WIDTH, IMG_HEIGHT = 224, 224
model = densenet121()
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 1)
model.load_state_dict(torch.load("C:/Users/User/OneDrive/Desktop/UM/WQF7009 XAI/AA/AA Pt2/densenet121_pneumonia.pth"))

# Initialize Grad-CAM with the loaded model
target_layer_name = "features.denseblock4.denselayer16.conv2"
grad_cam = GradCAM(model, target_layer_name)

# Function to run LRP
def lrp(image_tensor, target_class):
    composite = EpsilonGammaBox(low=-3., high=3., layer_map=[(BatchNorm, Epsilon())])

    test = image_tensor.clone().requires_grad_(True)
    target = torch.tensor(target_class).view(1,1)
    with Gradient(model, composite) as attributor:
        _, relevance = attributor(test, target)

    relevance_map = imgify(relevance[0].cpu().detach().sum(0), cmap='bwr', symmetric=True)

    # Convert the input tensor to a PIL image (normalize to [0, 255] if needed)
    image_pil = to_pil_image((image_tensor.squeeze(0) * 0.5 + 0.5))  # Assuming input normalized to [-1, 1]

    alpha = 0.2
    alpha_mask = Image.new("L", image_pil.size, int(255 * alpha))
    image_pil.putalpha(alpha_mask)

    return Image.alpha_composite(relevance_map.convert("RGBA"), image_pil.convert("RGBA")).convert("RGB")


# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = "/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            print(f"Saving file to: {file_path}")  # Debugging: Check file save location
            file.save(file_path)
            
            # Generate image tensor
            grad_cam_input = grad_cam.generate_image_tensor(file_path)

            # Grad-CAM heatmap
            heatmap = grad_cam.generate_heatmap(grad_cam_input)
            visualization = grad_cam.visualize_gradcam(grad_cam_input, heatmap)
            grad_cam_image = os.path.join(app.config["UPLOAD_FOLDER"], f"gradcam_{filename}")
            plt.imsave(grad_cam_image, visualization, cmap="jet")

            # LRP explanation
            output = torch.sigmoid(model(grad_cam_input))
            target_class = (output > 0.5).int()  # Binary classification (0 NORMAL, 1 PNEUMONIA)
            predicted = "PNEUMONIA" if target_class == 1 else "NORMAL"
            lrp_image = os.path.join(app.config["UPLOAD_FOLDER"], f"lrp_{filename}")
            lrp_explanation = lrp(grad_cam_input, target_class)
            lrp_explanation.save(lrp_image)

            # Render template with the images
            return render_template(
                "index.html", 
                image_url=filename, 
                gradcam_url=f"gradcam_{filename}", 
                lrp_url=f"lrp_{filename}",
                predicted=predicted,
                confidence = round(output.item(), 4) * 100 if target_class == 1 else round(1 - output.item(), 4) * 100
            )
    
    return render_template("index.html", image_url=None)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)

