import os
import cv2
import imutils
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense

app = FastAPI(title="Brain Tumor Detection API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model globally
def build_model():
    model = Sequential([
        ZeroPadding2D((2, 2), input_shape=(240, 240, 3)),
        Conv2D(32, (7, 7), strides=(1, 1)),
        BatchNormalization(axis=3),
        Activation('relu'),
        MaxPooling2D((4, 4)),
        MaxPooling2D((4, 4)),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

MODEL_PATH = "models/model.h5"
print(f"Loading model weights from {MODEL_PATH}...")
model = build_model()
model.load_weights(MODEL_PATH)
print("Model loaded successfully!")

def crop_brain_contour(image):
    """
    Finds the contours of the brain and crops the black space around it.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions + 
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if not cnts: # In case no contours are found, return original image
        return image
        
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # Crop new image out of the original image
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    
    return new_image

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read the file's content
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
             raise HTTPException(status_code=400, detail="Failed to decode the image.")

        # Preprocess the image
        img = crop_brain_contour(img)
        img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0) # Add batch dimension

        # Make prediction
        prediction = model.predict(img)
        prob = float(prediction[0][0])
        
        has_tumor = prob > 0.5
        
        # Result mapping
        # As per the normal binary classification layout:
        # Usually, class 1 is YES (Tumor), class 0 is NO. Let's return logic
        
        return {
            "has_tumor": has_tumor,
            "confidence": prob if has_tumor else 1 - prob,
            "tumor_probability": prob
        }
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
