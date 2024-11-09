import cv2
from API.Backend.ocr import extract_text
from API.Backend.model import get_model
from API.Backend.donut_extraction import donut_extraction
import tempfile

def process_image(cv2_img, yolo_model, donut_model):
    """
    Process an image using YOLO and Donut models for object detection and text extraction.

    Args:
        cv2_img (numpy.ndarray): The input image in OpenCV format.
        yolo_model: The YOLO model for object detection.
        donut_model: The Donut model for text extraction.

    Returns:
        tuple: A tuple containing:
            - image_with_boxes (numpy.ndarray): The input image with bounding boxes drawn.
            - paragraph_texts (dict): Extracted text from paragraph regions.
            - table_texts (dict): Extracted text from table regions.
            - donut_results: Results from the Donut model extraction.
    """
    # Perform inference with YOLO model
    results = yolo_model(cv2_img)

    # Extract bounding boxes, classes, and labels
    boxes = results[0].boxes.xyxy.numpy()
    classes = results[0].boxes.cls.numpy()
    names = results[0].names

    # Convert class indices to class names
    labels = [names[int(cls)] for cls in classes]

    # Filter and enumerate bounding boxes for 'Paragraph' and 'Table'
    paragraph_boxes = [(i+1, box) for i, (box, label) in enumerate(zip(boxes, labels)) if label == 'Paragraph']
    table_boxes = [(i+1, box) for i, (box, label) in enumerate(zip(boxes, labels)) if label == 'Table']

    # Draw boxes on the image
    image_with_boxes = draw_boxes(cv2_img, paragraph_boxes, table_boxes)

    # Extract text for paragraphs and tables
    paragraph_texts = extract_text(cv2_img, paragraph_boxes)
    table_texts = extract_text(cv2_img, table_boxes)

    # Add Donut extraction
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        cv2.imwrite(temp_file.name, cv2_img)
        donut_results = donut_extraction(temp_file.name, donut_model)

    return image_with_boxes, paragraph_texts, table_texts, donut_results

def draw_boxes(cv2_img, paragraph_boxes, table_boxes):
    """
    Draw bounding boxes and labels on the input image for paragraphs and tables.

    Args:
        cv2_img (numpy.ndarray): The input image in OpenCV format.
        paragraph_boxes (list): A list of tuples, each containing a number and a bounding box for paragraphs.
        table_boxes (list): A list of tuples, each containing a number and a bounding box for tables.

    Returns:
        numpy.ndarray: A copy of the input image with bounding boxes and labels drawn.

    Note:
        Paragraph boxes are drawn in blue, and table boxes are drawn in green.
        Each box is labeled with its type (Paragraph or Table) and a number.
    """
    image_with_boxes = cv2_img.copy()
    for number, box in paragraph_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image_with_boxes, f'Paragraph {number}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    for number, box in table_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_with_boxes, f'Table {number}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image_with_boxes
