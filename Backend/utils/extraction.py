import cv2 as cv
import os

def extract_components(image_path, model, save_dir='E:/Document Verfication/temp_images'):
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Predict components using the model
    results = model.predict(image_path)
    
    # Read the image in grayscale
    image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
    cropped_objects = {}

    # Loop through the results and crop the objects
    for i, box in enumerate(results[0].boxes):
        # Get the coordinates of the bounding box
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        cls = int(box.cls[0])  # Class index
        
        # Crop the detected region from the image
        cropped_obj = image[y1:y2, x1:x2]
        
        # Map the class name to the cropped object
        class_name = model.names[cls]
        
        # Print the class name and the shape of the cropped object
        print(f"Class: {class_name}, Shape: {cropped_obj.shape}")
        
        # Save the cropped object as a .png file
        cropped_image_filename = f"{class_name}_{i}.png"  # Naming based on class and index
        cropped_image_path = os.path.join(save_dir, cropped_image_filename)
        
        # Save the cropped image
        cv.imwrite(cropped_image_path, cropped_obj)
        print(f"Saved cropped image: {cropped_image_path}")

        cropped_objects[class_name] = cropped_image_path

    return cropped_objects
