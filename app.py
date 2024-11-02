from flask import Flask, request, render_template, jsonify
import numpy as np
import torch
from PIL import Image
import io
from torchvision import transforms
import snntorch as snn
import torch
import torch.nn as nn
import base64
from io import BytesIO
import os

app = Flask(__name__)

# Define the paths to each model state dictionary
MODEL_PATHS = {
    "IF": "models/if_model_state.pth",
    "LIF": "models/lif_model_state.pth",
    "IFLIF": "models/iflif_model_state.pth",
    "LIFIF": "models/lifif_model_state.pth"
}

# Define model configurations based on your saved models
MODEL_CONFIGS = {
    "IF": {"num_hidden": 4096, "num_steps": 30, "beta_lapicque": 0.90, "beta_leaky": 0},
    "LIF": {"num_hidden": 1024, "num_steps": 20, "beta_lapicque": 0, "beta_leaky": 0.95},
    "IFLIF": {"num_hidden": 512, "num_steps": 30, "beta_lapicque": 0.90, "beta_leaky": 0.90},
    "LIFIF": {"num_hidden": 1024, "num_steps": 40, "beta_lapicque": 0.95, "beta_leaky": 0.95}
}

# Define your model architecture (EnhancedSNN)
class EnhancedSNN(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden, num_steps, model_type, beta_lapicque=0, beta_leaky=0):
        super(EnhancedSNN, self).__init__()

        # Define layers based on model_type to match the original training code
        if model_type == 'LIF':
            self.layer1 = snn.Leaky(beta=beta_leaky)  # Leaky Integrate-and-Fire layer
            self.layer2 = snn.Leaky(beta=beta_leaky)  # Leaky Integrate-and-Fire layer
        elif model_type == 'IF':
            self.layer1 = snn.Lapicque(beta=beta_lapicque)  # Integrate-and-Fire layer
            self.layer2 = snn.Lapicque(beta=beta_lapicque)  # Integrate-and-Fire layer
        elif model_type == 'IFLIF':
            self.layer1 = snn.Lapicque(beta=beta_lapicque)  # Integrate-and-Fire layer
            self.layer2 = snn.Leaky(beta=beta_leaky)       # Leaky Integrate-and-Fire layer
        elif model_type == 'LIFIF':
            self.layer1 = snn.Leaky(beta=beta_leaky)       # Leaky Integrate-and-Fire layer
            self.layer2 = snn.Lapicque(beta=beta_lapicque) # Integrate-and-Fire layer

        # Fully connected layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_outputs)

        # Number of timesteps for spiking neurons
        self.num_steps = num_steps

    def forward(self, x):
        # Initialize potentials based on the layer type
        potential1 = self.layer1.init_lapicque() if isinstance(self.layer1, snn.Lapicque) else self.layer1.init_leaky()
        potential2 = self.layer2.init_lapicque() if isinstance(self.layer2, snn.Lapicque) else self.layer2.init_leaky()

        # Store spikes at each timestep
        spikes1, spikes2 = [], []

        # Loop through timesteps
        for step in range(self.num_steps):
            # Flatten the input for the fully connected layer
            current1 = self.fc1(x.view(x.size(0), -1))

            # First spiking layer
            spike1, potential1 = self.layer1(current1, potential1)

            # Second fully connected layer
            current2 = self.fc2(spike1)

            # Second spiking layer
            spike2, potential2 = self.layer2(current2, potential2)

            # Collect spikes for each timestep
            spikes1.append(spike1)
            spikes2.append(spike2)

        # Stack spikes to return them as outputs over timesteps
        return torch.stack(spikes1, dim=0), torch.stack(spikes2, dim=0)

# Function to load model based on user selection
def load_model(model_choice):
    config = MODEL_CONFIGS[model_choice]
    num_inputs = 28 * 28  # For MNIST image size
    num_outputs = 10      # For MNIST classes (0-9)

    # Initialize model with matching configuration
    model = EnhancedSNN(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        num_hidden=config["num_hidden"],
        num_steps=config["num_steps"],
        model_type=model_choice,
        beta_lapicque=config["beta_lapicque"],
        beta_leaky=config["beta_leaky"]
    )
    
    # Load the saved state dictionary
    model.load_state_dict(torch.load(MODEL_PATHS[model_choice], map_location=torch.device('cpu')))
    model.eval()
    return model


def preprocess_image(image, rotation_range):
    # Random rotation within the specified range
    rotation_angle = np.random.uniform(-rotation_range, rotation_range)
    rotated_image = transforms.functional.rotate(image, rotation_angle)

    # Convert to tensor and normalize to match the training setup
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),  # Scales pixel values to [0, 1]
    ])
    processed_image = transform(rotated_image).unsqueeze(0)  # Add batch dimension

    return processed_image, rotated_image, rotation_angle

# Helper function to encode image for display in HTML
def encode_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form.get("model_choice")
    rotation = int(request.form.get("rotation"))
    file = request.files.get('file')

    if not file or model_choice not in MODEL_PATHS:
        return jsonify({"error": "Invalid input"}), 400

    # Load the selected model
    model = load_model(model_choice)

    # Open the uploaded image
    image = Image.open(io.BytesIO(file.read()))
    
    # Preprocess the image: get both the rotated tensor and rotated image for display
    processed_image, rotated_image, rotation_angle = preprocess_image(image, rotation)

    # Run the model on the processed image
    with torch.no_grad():
        _, output = model(processed_image)
        prediction = output.sum(dim=0).argmax(dim=1).item()

    # Encode images for HTML display
    original_image_url = encode_image(image)
    rotated_image_url = encode_image(rotated_image)

    return jsonify({
        "prediction": prediction,
        "rotation_angle": rotation_angle,
        "original_image_url": original_image_url,
        "rotated_image_url": rotated_image_url
    })



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # app.run(host= "0.0.0.0", port= port, debug=True)
    app.run(host= "0.0.0.0", port= port)
