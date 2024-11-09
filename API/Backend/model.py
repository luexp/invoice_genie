from ultralytics import YOLO

_model = None

def get_model():
    """
    Get or initialize the YOLO model.

    This function implements a singleton pattern for the YOLO model.
    It ensures that only one instance of the model is created and reused.

    Returns:
        YOLO: An instance of the YOLO model.

    Note:
        The model is loaded from the file "API/Backend/best.pt" when first called.
    """
    global _model
    if _model is None:
        _model = YOLO("API/Backend/best.pt")
    return _model
