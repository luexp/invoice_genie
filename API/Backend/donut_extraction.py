import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image

def load_donut_model():
    """
    Load and initialize the Donut model for text extraction.

    Returns:
        tuple: A tuple containing the Donut processor and model.
    """
    processor = DonutProcessor.from_pretrained("to-be/donut-base-finetuned-invoices")
    model = VisionEncoderDecoderModel.from_pretrained("to-be/donut-base-finetuned-invoices")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model

def donut_extraction(image_filename, donut_model):
    """
    Perform text extraction on an image using the Donut model.

    Args:
        image_filename (str): The filename of the image to process.
        donut_model (tuple): A tuple containing the Donut processor and model.

    Returns:
        dict or None: A dictionary containing the extracted text information,
                      or None if an error occurs during processing.
    """
    processor, model = donut_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    try:
        image = Image.open(image_filename).convert("RGB")
    except IOError:
        print(f"Error: Unable to open image file {image_filename}")
        return None

    task_prompt = "<s_text_extraction>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    pixel_values = processor(image, return_tensors="pt").pixel_values

    try:
        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        result = processor.token2json(sequence)
        return result
    except Exception as e:
        print(f"Error during model inference or processing: {str(e)}")
        return None
