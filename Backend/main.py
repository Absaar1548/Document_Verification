from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import easyocr
import torch
import os
from Backend.utils.extraction import extract_components
from Backend.utils.signature_extraction import clean_signature , match_signature
from Backend.utils.load_models import load_verfication_model, load_cleaning_model
import cv2 as cv
from PIL import Image
import io
import re
import pytesseract

# Initialize FastAPI app
app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Declare global variables for all models
models = {
    "yolo": None,
    "ocr": None,
    "verification": None,
    "signature_cleaning_pipeline": None,
}

@app.on_event("startup")
async def load_models():
    """
    Load all models during FastAPI startup event and ensure they are on GPU.
    """
    global models

    # Check if GPU is available
    print(f"Device selected: {device}")

    # Load YOLO model on GPU
    models["yolo"] = YOLO(r"E:\Document Verfication\Models\Extraction\weights\best.pt")  # Replace with your trained YOLO weights
    models["yolo"].to(device)  # Move YOLO model to GPU
    print("YOLO model loaded successfully on GPU!")

    # Load EasyOCR model on GPU
    models["ocr"] = easyocr.Reader(["en"], gpu=torch.cuda.is_available())  # Enable GPU if available
    print("EasyOCR model loaded successfully on GPU!")

    # Load signature cleaning pipeline model on GPU
    # models["signature_cleaning_pipeline"] = torch.load("models/signature_cleaning_model.pt")
    # models["signature_cleaning_pipeline"].to(device)  # Move to GPU
    print("Signature cleaning pipeline model loaded successfully on GPU!")

    # Load verification model on GPU
    # Replace with your verification model's loading logic
    MODEL_NAME = "shufflenet"  # Example, change based on your model
    MODEL_WEIGHTS_PATH = r"E:\Document Verfication\Models\Verfication\models\shufflenet_best.pth"
    models["verification"] = load_verfication_model(model_name=MODEL_NAME,model_weights=
                        MODEL_WEIGHTS_PATH,device= device)
    print("Verification model loaded successfully on GPU!")

@app.post("/verify-document/")
async def verify_document(file: UploadFile = File(...)):
    """
    Endpoint to verify the document, extract components, and return verification results.
    """
    try:
        # Step 1: Read and save the uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        temp_path = "E:/Document Verfication/temp_images/temp_uploaded_image.png"
        image.save(temp_path, format="PNG")

        # Step 2: Extract components using YOLO
        extracted_images = extract_components(temp_path, models["yolo"])
        if "account" not in extracted_images or "signature" not in extracted_images:
            return JSONResponse(content={"error": "Failed to extract account_no or signature"}, status_code=400)
        account_no_image = extracted_images["account"]
        signature_image = extracted_images["signature"]
        print("Compnents Retreived")

        # Step 3: Process account_no and signature

        ## OCR for account_no
        
        account_img = cv.imread(account_no_image)  # Using OpenCV to read the image
        # account_no = pytesseract.image_to_string(account_img, config='outputbase digits')
        account_no = models["ocr"].readtext(account_img, detail=0)[0]  # Simplified OCR usage
        account_no = re.sub(r"[^\d]", "", account_no)

        print("OCR Completed")
        print("Account Number: ", account_no)

        ## Cleaning pipeline for signature
        cleaned_signature = clean_signature(signature_image, models["signature_cleaning_pipeline"])
        print("Cleaning Completed")

        # Step 4: Match the signature
        folder_path = f"E:/Document Verfication/Database/{account_no}/"
        if not os.path.exists(folder_path):
            return JSONResponse(content={"error": f"Account folder for {account_no} not found in database"}, status_code=404)
        verification_status, distance = match_signature(cleaned_signature, folder_path, models["verification"] , device=device)
        print("Verification Completed")

        # Step 5: Return JSON response
        response = {
            "account_details": account_no,
            "signature": cleaned_signature,  # Replace with actual cleaned signature path if saved
            "Verification details": "Matched" if verification_status else "Not Matched",
            "Distance": distance,
        }
        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
