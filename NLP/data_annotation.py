import google.generativeai as genai
import os
import json
import time
from typing import List, Dict
import tenacity

"""
This module provides functionality for annotating invoice data using Google's Generative AI model.

The module includes functions for:
1. Configuring the API and setting up the generative model
2. Processing text data in batches to extract entities
3. Reviewing and editing annotations
4. Preparing the annotated data for training a Named Entity Recognition (NER) model

Key functions:
- configure_api(): Sets up the Google Generative AI model
- generate_with_retry(): Generates content with retry logic
- batch_process(): Processes texts in batches to extract entities
- review_annotations(): Allows manual review and editing of annotations
- prepare_training_data(): Prepares the final training data for NER
- load_and_annotate_data(): Main function to load, annotate, and prepare training data

The module uses error handling, rate limiting, and retry logic to ensure robust processing of large datasets.
"""

def configure_api():
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    return genai.GenerativeModel('gemini-pro')

@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
def generate_with_retry(model, prompt):
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        print(f"Error generating content: {e}")
        raise

def batch_process(model, texts: List[str], batch_size: int = 10) -> List[Dict[str, str]]:
    all_annotations = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        prompt = "Process the following invoices and extract entities:\n\n"
        prompt += "\n---\n".join(batch)
        prompt += "\n\nRespond with a list of JSON objects, one for each invoice."

        try:
            batch_annotations = generate_with_retry(model, prompt)
            all_annotations.extend(batch_annotations)
        except Exception as e:
            print(f"Failed to process batch starting at index {i}: {e}")

        time.sleep(1)  # Rate limiting

    return all_annotations

def review_annotations(text: str, annotations: Dict[str, str]) -> Dict[str, str]:
    print(f"Text: {text}")
    print("Annotations:")
    for entity, value in annotations.items():
        print(f"  {entity}: {value}")

    while True:
        action = input("Enter 'a' to accept, 'e' to edit, or 'r' to reject: ").lower()
        if action == 'a':
            return annotations
        elif action == 'e':
            entity = input("Enter entity to edit: ")
            value = input("Enter new value: ")
            annotations[entity] = value
        elif action == 'r':
            return None
        else:
            print("Invalid input. Please try again.")

def prepare_training_data(texts: List[str], annotations: List[Dict[str, str]]) -> List[tuple]:
    training_data = []
    for text, anno in zip(texts, annotations):
        reviewed_anno = review_annotations(text, anno)
        if reviewed_anno:
            entities = [(text.index(v), text.index(v) + len(v), k) for k, v in reviewed_anno.items()]
            training_data.append((text, {"entities": entities}))
    return training_data

def load_and_annotate_data(data_path: str) -> List[tuple]:
    model = configure_api()

    with open(data_path, 'r') as f:
        texts = [line.strip() for line in f]

    raw_annotations = batch_process(model, texts)
    return prepare_training_data(texts, raw_annotations)
