import os
from flask import Flask, request, render_template, jsonify
import torch
from torchvision import transforms
from PIL import Image
from model import SkinLesionModel  # Assuming your model is in a file named model.py

app = Flask(__name__)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkinLesionModel(num_classes=2).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            image = Image.open(file.stream).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)

            result = "Melanoma" if predicted.item() == 1 else "Non-Melanoma"
            return jsonify({'result': result})
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)