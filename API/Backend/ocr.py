import easyocr
import cv2

def extract_text(cv2_img, numbered_boxes):
    """
    Extract text from specified regions of an image using EasyOCR.

    Args:
        cv2_img (numpy.ndarray): The input image in OpenCV format.
        numbered_boxes (list): A list of tuples, each containing a number and a bounding box.
                               The bounding box should be in the format (x_min, y_min, x_max, y_max).

    Returns:
        dict: A dictionary where keys are the numbers associated with each bounding box,
              and values are lists of dictionaries containing the extracted text and confidence scores.

    Example:
        >>> img = cv2.imread('image.jpg')
        >>> boxes = [(1, (10, 10, 100, 50)), (2, (150, 150, 300, 200))]
        >>> result = extract_text(img, boxes)
    """
    reader = easyocr.Reader(['fr', 'es', 'en'])
    extracted_texts = {}
    for number, bbox in numbered_boxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        cropped_image = cv2_img[y_min:y_max, x_min:x_max]
        ocr_results = reader.readtext(cropped_image)
        box_text = [{"text": text, "confidence": prob} for (_, text, prob) in ocr_results]
        extracted_texts[number] = box_text
    return extracted_texts
