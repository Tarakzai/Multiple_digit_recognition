import os
import json
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# Set random seed 
torch.manual_seed(42)

# Function to load the MNIST dataset
def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return train_dataset

# Function to combine images horizontally
def combine_images(images, max_digits=5):
    combined_image = Image.new('L', (28 * max_digits, 28), color=255)
    
    for i in range(min(len(images), max_digits)):
        combined_image.paste(images[i], (28 * i, 0))
    
    return combined_image

# Create the data folder
data_folder = 'train_data'
combined_images_folder = os.path.join(data_folder, 'combined_images')
os.makedirs(combined_images_folder, exist_ok=True)

# Load the MNIST dataset
mnist_dataset = load_mnist()
data_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=5, shuffle=True)


labels_dict = {}

# Iterate through the dataset and generate combined images and labels
for batch_idx, (data, target) in enumerate(data_loader):
    # Convert tensor to numpy array and then to PIL Image
    images = [transforms.ToPILImage()(img) for img in data]

    # Combine images horizontally
    combined_image = combine_images(images)

    # Save the combined image in the combined_images_folder
    img_filename = f'{batch_idx + 1:04d}.png'
    img_path = os.path.join(combined_images_folder, img_filename)
    combined_image.save(img_path)

    # Create label and store it in the dictionary
    label = [int(digit) for digit in target.numpy()]
    labels_dict[img_filename] = label

# Save the labels in JSON format in the data_folder
json_filename = 'labels.json'
json_path = os.path.join(data_folder, json_filename)
with open(json_path, 'w') as json_file:
    json.dump(labels_dict, json_file)

print(f"Combined images saved in '{combined_images_folder}'.")
print(f"Labels saved in '{json_path}'.")

