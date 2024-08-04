import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
from io import BytesIO

# Load the pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)
model.eval()

# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to get the image from a URL
def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

# Function to predict the dog breed
def predict_dog_breed(url):
    img = load_image(url)
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Load ImageNet labels
    LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels = requests.get(LABELS_URL).json()
    
    _, predicted_idx = torch.max(outputs, 1)
    breed = labels[predicted_idx.item()]
    
    return breed

# Example usage
image_url = 'https://example.com/dog.jpg'  # Replace with your image URL
breed = predict_dog_breed(image_url)
print(f'The predicted dog breed is: {breed}')
