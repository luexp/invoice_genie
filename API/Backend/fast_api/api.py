from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from API.Backend.model import get_model
from API.Backend.donut_extraction import load_donut_model
from API.Backend.image_processing import process_image
from API.Backend.file_utils import handle_file_upload, setup_temp_directory, cleanup_old_images
from API.Backend.config import TEMP_IMAGE_DIR
import uuid
import cv2

app = FastAPI()


app.state.yolo_model = get_model()
app.state.donut_model = load_donut_model()
# # Allow all requests (optional, good for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Setup and mount temp directory
setup_temp_directory(app)

@app.get("/")
def index():
    return {"status": "ok"}


@app.post('/upload_invoice')
async def receive_file(file: UploadFile = File(...)):
    cv2_img = await handle_file_upload(file)
    image_with_boxes, paragraph_texts, table_texts, donut_results = process_image(cv2_img, app.state.yolo_model, app.state.donut_model)
    # Save the image with bounding boxes
    unique_filename = f"{uuid.uuid4()}.png"
    image_path = os.path.join(TEMP_IMAGE_DIR, unique_filename)
    cv2.imwrite(image_path, image_with_boxes)

    # Prepare the response
    response = {
        "paragraphs": paragraph_texts,
        "tables": table_texts,
        "image_url": f"/temp_images/{unique_filename}",
        "donut_extraction": donut_results
    }

    return JSONResponse(content=response)

# Cleanup task
app.on_event("startup")(cleanup_old_images)
app.on_event("shutdown")(cleanup_old_images)
