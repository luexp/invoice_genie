import os
import time
import uuid
import cv2
import numpy as np
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from pdf2image import convert_from_bytes
import io
from API.Backend.config import TEMP_IMAGE_DIR

def setup_temp_directory(app):
    """
    Set up a temporary directory for storing images and mount it to the FastAPI app.

    Args:
        app: The FastAPI application instance.

    Returns:
        None
    """
    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
    app.mount("/temp_images", StaticFiles(directory=TEMP_IMAGE_DIR), name="temp_images")

async def handle_file_upload(file):
    """
    Handle file upload for images and PDFs.

    Args:
        file: The uploaded file object.

    Returns:
        numpy.ndarray: The image as a NumPy array in OpenCV format.

    Raises:
        HTTPException: If the file format is unsupported.
    """
    contents = await file.read()
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension in ['.jpg', '.jpeg', '.png']:
        return cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    elif file_extension == '.pdf':
        return convert_pdf_to_image(contents)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

def convert_pdf_to_image(contents):
    """
    Convert the first page of a PDF to an image.

    Args:
        contents (bytes): The PDF file contents.

    Returns:
        numpy.ndarray: The image as a NumPy array in OpenCV format.

    Raises:
        HTTPException: If the PDF conversion fails.
    """
    images = convert_from_bytes(contents, first_page=1, last_page=1)
    if not images:
        raise HTTPException(status_code=400, detail="Failed to convert PDF to image")
    img_byte_arr = io.BytesIO()
    images[0].save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return cv2.imdecode(np.frombuffer(img_byte_arr, np.uint8), cv2.IMREAD_COLOR)

async def cleanup_old_images():
    """
    Clean up old temporary images that are more than an hour old.

    This function is intended to be run periodically to remove old temporary files.

    Returns:
        None
    """
    current_time = time.time()
    for filename in os.listdir(TEMP_IMAGE_DIR):
        file_path = os.path.join(TEMP_IMAGE_DIR, filename)
        try:
            if os.path.isfile(file_path) and os.path.getmtime(file_path) < current_time - 3600:
                os.remove(file_path)
                print(f"Removed old temporary file: {filename}")
        except Exception as e:
            print(f"Error removing file {filename}: {str(e)}")
    print("Cleanup of old temporary images completed")
