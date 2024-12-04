import os
from PIL import Image, ImageOps
import torch
<<<<<<< HEAD
from torchvision.transforms import v2, InterpolationMode
from Models.Verfication.src.models.cnn import MODELS
import cv2 as cv

transforms = [
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(232, interpolation=InterpolationMode.BICUBIC, antialias=True),
    v2.CenterCrop(224),
    v2.Normalize(mean=[0.5], std=[0.5])
]

transforms = v2.Compose(transforms)
=======
import cv2 as cv
from torchvision.transforms import functional as F
from Models.Verfication.src.models.cnn import MODELS
from torchvision.transforms import v2, InterpolationMode

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((232,232), interpolation=InterpolationMode.BICUBIC),
    v2.CenterCrop(224),
    v2.Normalize(mean=[0.5], std=[0.5])
])
>>>>>>> 98d4f7bd5dfb7507ccf16f1d3a8c0abd8888d2ad

def preprocess_image(image_path: str, resize: tuple = (224, 224)):
    """
    Preprocesses an image by inverting, converting to grayscale, resizing,
    and normalizing.

    Args:
    - image_path (str): Path to the image.
    - resize (tuple): The target size to resize the image to.

    Returns:
    - torch.Tensor: Processed image tensor.
    """
    try:
        # Load and invert the image
<<<<<<< HEAD
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        image = cv.threshold(image, 127, 255, cv.THRESH_BINARY_INV)[1]
        image = transforms(image)
        return image
    
=======
        # image = Image.open(image_path).convert("L")
        # image = ImageOps.invert(image)

        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        image = cv.threshold(image, 127, 255, cv.THRESH_BINARY_INV)[1]

        # Apply transformations: resize and normalize
        return transform(image)
>>>>>>> 98d4f7bd5dfb7507ccf16f1d3a8c0abd8888d2ad
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def predict(img1: str, img2: str, model, device: str = 'cpu'):
    """
    Predict the similarity between two signature images using the provided model.

    Args:
    - img1 (str): Path to the first image (signature).
    - img2 (str): Path to the second image (signature).
    - model: The model used for similarity calculation.
    - device (str): Device to run the model ('cpu' or 'cuda').

    Returns:
    - torch.Tensor: The distance between the two images (similarity score).
    """
    print(f"Starting prediction for images '{img1}' and '{img2}'")

    # Preprocess the images
    image1 = preprocess_image(img1)
    image2 = preprocess_image(img2)

    if image1 is None or image2 is None:
        raise ValueError(f"One or both of the images '{img1}' or '{img2}' couldn't be processed.")

    # Move images to the correct device
    image1 = image1.unsqueeze(0).to(device)
    image2 = image2.unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        print("Passing images through the model for distance calculation")
        distance = model(image1, image2)
        print("Distance calculation completed")

    return distance

def match_signature(cleaned_signature: str, folder_path: str, verification_model, device: str = 'cpu', distance_threshold: float = 0.242):
    """
    Match the cleaned signature with original signatures in the given folder using the verification model.

    Args:
    - cleaned_signature (str): Path to the cleaned signature image to be verified.
    - folder_path (str): Path to the folder containing original signature images.
    - verification_model: The pre-trained model used for signature comparison.
    - device (str): Device to run the model on ('cpu' or 'cuda').
    - distance_threshold (float): Threshold for considering signatures as matched.

    Returns:
    - verification_status (bool): True if signatures match, False otherwise.
    - average_distance (float): The average distance between the cleaned signature and all original signatures.
    """
    print(f"Starting signature verification for cleaned signature: {cleaned_signature}")
    
    # List to store distances for each comparison
    distances = []

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Case insensitive file extension matching
            original_signature_path = os.path.join(folder_path, file_name)
            
            # Use the predict function to get the distance between the cleaned signature and the current original signature
            try:
                distance = predict(cleaned_signature, original_signature_path, verification_model, device)
                distances.append(distance.item())  # Append the distance value to the list
            except ValueError as e:
                print(e)
                continue  # Skip this file if there was an error

    if not distances:
        print("No valid distances calculated. Check the input files or preprocessing.")
        return False, 0.0

    # Calculate majority vote for matching: distance < threshold means a match
    matches = [dist < distance_threshold for dist in distances]

    # The final verification status is True if most distances are below the threshold
    verification_status = sum(matches) > len(matches) / 2  # Majority voting rule
    
    # Calculate the average distance
    average_distance = sum(distances) / len(distances)
    
    # Return the verification status and the average distance
    return verification_status, average_distance

def clean_signature(signature_image, cleaning_pipeline_model):
    return signature_image