import spacy
from spacy.tokens import DocBin
from spacy.util import minibatch, compounding
import random

def train_ner(train_data, output_dir, n_iter=50):
    """
    Train a Named Entity Recognition (NER) model using spaCy.

    This function loads a pre-trained French language model, adds or gets the NER pipeline,
    and trains it on the provided data. The training process uses minibatching and compounding
    batch sizes for efficiency. The trained model is then saved to disk.

    Parameters:
    train_data (list): A list of tuples, where each tuple contains (text, annotations) pairs.
    output_dir (str): The directory path where the trained model will be saved.
    n_iter (int, optional): The number of training iterations. Defaults to 50.

    The function prints the losses for each iteration during training and a completion message
    when finished.

    Note:
    - The function uses a French language model ("fr_core_news_md").
    - It adds labels for INVOICE_NUMBER, DATE, TOTAL_AMOUNT, COMPANY_NAME, and ADDRESS.
    - The training process uses a dropout of 0.35 and shuffles the training data in each iteration.
    """
    nlp = spacy.load("fr_core_news_md")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for label in LABELS:
        ner.add_label(label)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print(f"Losses at iteration {itn}: {losses}")

    nlp.to_disk(output_dir)
    print("NER model training completed and saved.")

# Constants
LABELS = ["INVOICE_NUMBER", "DATE", "TOTAL_AMOUNT", "COMPANY_NAME", "ADDRESS"]
