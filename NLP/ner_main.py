from data_annotation import load_and_annotate_data
from ner_training import train_ner

def main():
    data_path = "path_to_your_training_data"
    output_dir = "./ner_model"

    # Load and annotate data
    training_data = load_and_annotate_data(data_path)

    # Train NER model
    train_ner(training_data, output_dir)

if __name__ == "__main__":
    main()
