import sys
import os
import json
from transformers import BertTokenizer
import random
import tqdm

BERT_VOCAB = "bert-base-uncased"
MAX_SEQ_LENGTH = 128

# Set a fixed random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


from tqdm import tqdm

def write_in_hsln_format(input, hsln_format_txt_dirpath, tokenizer, k_min=1, k_max=3, context_type="left", randomize=False):
    """
    Prepares the data in HSLN-compatible format using a fixed or randomized local context.

    :param input: Input data (JSON).
    :param hsln_format_txt_dirpath: Path to the output file.
    :param tokenizer: BERT tokenizer.
    :param k_min: Minimum number of context sentences.
    :param k_max: Maximum number of context sentences.
    :param context_type: Type of context ('left', 'right', 'neighbors', or 'all').
    :param randomize: Whether to randomize the context size for each instance.
    """
    final_string = ''
    filename_sent_boundaries = {}

    # Add a tqdm progress bar for the outer loop
    for file in tqdm(input, desc="Processing files", unit="file"):
        file_name = file['id']
        annotations = file['annotations'][0]['result']
        sentence_count = len(annotations)
        filename_sent_boundaries[file_name] = {"sentence_span": []}

        # Add a tqdm progress bar for the inner loop
        for sentence_idx, annotation in enumerate(tqdm(annotations, desc=f"Processing sentences in {file_name}", unit="sentence", leave=False)):
            file_id = f"{file_name}/{sentence_idx}"
            final_string += f"###{file_id}\n"

            filename_sent_boundaries[file_name]['sentence_span'].append(
                [annotation['value']['start'], annotation['value']['end']]
            )

            # Determine context size
            if context_type == "neighbors":
                # Ensure k is even and randomize only for train/dev/test
                if randomize:
                    k = random.choice([x for x in range(k_min, k_max + 1) if x % 2 == 0])
                else:
                    k = k_max

                left_k = k // 2
                right_k = k // 2

                # Extract left and right contexts
                left_context = annotations[max(0, sentence_idx - left_k):sentence_idx]
                right_context = annotations[sentence_idx + 1:min(sentence_count, sentence_idx + 1 + right_k)]

                # Combine left and right contexts with the target sentence in the middle
                context_sentences = left_context + [annotations[sentence_idx]] + right_context
            elif context_type == "left":
                k = random.randint(k_min, k_max) if randomize else k_max
                start_idx = max(0, sentence_idx - k)
                context_sentences = annotations[start_idx:sentence_idx + 1]  # Include target sentence in its position
            elif context_type == "right":
                k = random.randint(k_min, k_max) if randomize else k_max
                end_idx = min(sentence_count, sentence_idx + k + 1)
                context_sentences = annotations[sentence_idx:end_idx]  # Include target sentence in its position
            elif context_type == "none":
                # Include only the target sentence
                context_sentences = [annotations[sentence_idx]]
            elif context_type == "all":
                # Take all sentences in the document as context
                context_sentences = annotations
            else:
                raise ValueError("Invalid context type. Choose 'left', 'right', 'neighbors', or 'all'.")

            for context_annotation in context_sentences:
                sentence_txt = context_annotation['value']['text']
                sentence_label = context_annotation['value']['labels'][0]
                sentence_txt = sentence_txt.replace("\r", "")

                if sentence_txt.strip():
                    sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True, max_length=MAX_SEQ_LENGTH)
                    sent_tokens = [str(i) for i in sent_tokens]
                    sent_tokens_txt = " ".join(sent_tokens)

                    # Target indicator is 1 for the target sentence, 0 otherwise
                    target_indicator = 1 if context_annotation == annotation else 0
                    final_string += f"{sentence_label}\t{sent_tokens_txt}\t{target_indicator}\n"

            final_string += "\n"

    with open(hsln_format_txt_dirpath, "w") as file:
        file.write(final_string)



def tokenize_and_save(input_json_path, output_dir, file_suffix, tokenizer, label_type, k_min=1, k_max=3, context_type="left"):
    """
    Tokenizes the input JSON data and saves it in HSLN format.

    Args:
    - input_json_path (str): Path to the input JSON file.
    - output_dir (str): Directory where the output file should be saved.
    - file_suffix (str): Suffix for the output file name (e.g., '_scibert').
    - tokenizer (BertTokenizer): BERT tokenizer instance.
    - label_type (str): The type of label to use (e.g., 'steps', 'category', 'rhetorical_function').
    - k_min (int): Minimum number of sentences before/after the target sentence to include as context.
    - k_max (int): Maximum number of sentences before/after the target sentence to include as context.
    - context_type (str): Type of context ('left', 'right', or 'neighbors').
    """
    context_dir = os.path.join(output_dir, context_type)
    os.makedirs(context_dir, exist_ok=True)

    input_data = json.load(open(input_json_path))

    # Randomized context size for train and dev splits
    randomize = "train" in file_suffix or "dev" in file_suffix or "test" in file_suffix
    output_file_path = os.path.join(context_dir, f"{file_suffix}_scibert.txt")
    write_in_hsln_format(input_data, output_file_path, tokenizer, k_min, k_max, context_type, randomize=randomize)

    print(f"Tokenized data for {label_type} with context type '{context_type}' and size ({k_min}-{k_max}) saved to: {output_file_path}")

    # Fixed context sizes for test split
    if "test" in file_suffix and context_type == "neighbors":
        for k in [2, 4, 6]:  # Only even values for neighbors
            output_file_path = os.path.join(context_dir, f"test_scibert_{k}.txt")
            write_in_hsln_format(input_data, output_file_path, tokenizer, k, k, context_type, randomize=False)
            print(f"Tokenized data for {label_type} with context type '{context_type}' and size {k} saved to: {output_file_path}")

    # Fixed context sizes for test split
    elif "test" in file_suffix and context_type in ["left", "right"]:
        for k in range(k_min, k_max + 1):
            output_file_path = os.path.join(context_dir, f"test_scibert_{k}.txt")
            write_in_hsln_format(input_data, output_file_path, tokenizer, k, k, context_type, randomize=False)
            print(f"Tokenized data for {label_type} with context type '{context_type}' and size {k} saved to: {output_file_path}")


def tokenize():
    """
    Main function to handle tokenization for 'steps', 'category', and 'rhetorical function'.
    """
    if len(sys.argv) != 11:
        print("Usage: python local_context.py --input_dir <input_dir> --output_dir <output_dir> --k_min <k_min> --k_max <k_max> --context_type <left|right|neighbors>")
        sys.exit(1)

    input_dir = sys.argv[2]
    output_dir = sys.argv[4]
    k_min = int(sys.argv[6])
    k_max = int(sys.argv[8])
    context_type = sys.argv[10]

    # Validate context type
    if context_type not in ["left", "right", "neighbors", "none", "all"]:
        print("Invalid context type. Choose 'left', 'right', or 'neighbors'.")
        sys.exit(1)

    # Files to process
    files = {
        "train": os.path.join(input_dir, "train.json"),
        "dev": os.path.join(input_dir, "dev.json"),
        "test": os.path.join(input_dir, "test.json")
    }

    tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)

    for split, filepath in files.items():
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        tokenize_and_save(filepath, output_dir, split, tokenizer, f"{context_type}_context", k_min, k_max, context_type)

# Entry point of the script
if __name__ == "__main__":
    tokenize()
