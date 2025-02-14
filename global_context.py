import sys
import os
import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer
import random
import spacy
import torch

from tqdm import tqdm



BERT_VOCAB = "./models/bert-base-uncased"
MAX_SEQ_LENGTH = 128
# Load Sentence-BERT model
sentencebert_model = SentenceTransformer('./models/all-mpnet-base-v2')  # You can change the model name as needed


# Set a fixed random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

import spacy

# Load SpaCy model for named entity recognition
nlp = spacy.load("en_core_web_sm")

def common_entities_retrieve(document, query_idx, k):
    """
    Retrieve the top-k sentences based on shared named entities with the target sentence.

    Args:
    - document (List[dict]): List of sentences in the document.
    - query_idx (int): Index of the target sentence.
    - k (int): Number of sentences to retrieve.

    Returns:
    - List[dict]: List of the top-k sentences that share common entities with the target sentence.
    """
    # Extract sentence texts
    sentences = [sent['value']['text'] for sent in document]

    # Perform NER on the query sentence and extract entities
    query_entities = {ent.text.lower() for ent in nlp(sentences[query_idx]).ents}

    if not query_entities:
        return []  # If no entities in the query sentence, return an empty list

    # Compute entity overlap scores for all sentences
    entity_overlap_scores = []
    for idx, sentence in enumerate(sentences):
        if idx == query_idx:
            continue  # Skip the query sentence itself

        sentence_entities = {ent.text.lower() for ent in nlp(sentence).ents}
        overlap = query_entities.intersection(sentence_entities)
        entity_overlap_scores.append((idx, len(overlap)))  # (sentence index, overlap score)

    # Sort by overlap score in descending order
    entity_overlap_scores = sorted(entity_overlap_scores, key=lambda x: x[1], reverse=True)

    # Get the top-k sentences
    top_k_indices = [idx for idx, _ in entity_overlap_scores[:k]]
    return [document[i] for i in top_k_indices]


def sentencebert_retrieve(document, query_idx, k):
    """
    Retrieve the top-k most similar sentences using Sentence-BERT.

    Args:
    - document (List[dict]): List of sentences in the document.
    - query_idx (int): Index of the target sentence.
    - k (int): Number of sentences to retrieve.

    Returns:
    - List[dict]: List of the top-k retrieved sentences.
    """
    # Extract sentence texts
    sentences = [sent['value']['text'] for sent in document]
    query = sentences[query_idx]

    # Compute embeddings for all sentences
    embeddings = sentencebert_model.encode(sentences, convert_to_tensor=True)
    query_embedding = embeddings[query_idx]

    # Compute cosine similarity
    cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]

    # Exclude the query sentence itself
    cosine_scores[query_idx] = -1

    # Get top-k indices
    top_k_indices = torch.topk(cosine_scores, k=min(k, len(cosine_scores))).indices.tolist()

    # Retrieve the sentences corresponding to the top-k indices
    return [document[i] for i in top_k_indices]


def bm25_retrieve(document, query_idx, k):
    """
    Retrieve the top-k most similar sentences using BM25.

    Args:
    - document (List[dict]): List of sentences in the document.
    - query_idx (int): Index of the target sentence.
    - k (int): Number of sentences to retrieve.

    Returns:
    - List[dict]: List of the top-k retrieved sentences.
    """
    bm25_model = BM25Okapi([sent['value']['text'].split() for sent in document])
    query = document[query_idx]['value']['text'].split()
    scores = bm25_model.get_scores(query)
    scores[query_idx] = -1  # Exclude the target sentence itself

    # Get top-k indices
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [document[i] for i in top_k_indices]

def random_retrieve(document, query_idx, k, sentence_idx, seed_k):
    """
    Retrieve k random sentences from the document, excluding the target sentence.

    Args:
    - document (List[dict]): List of sentences in the document.
    - query_idx (int): Index of the target sentence.
    - k (int): Number of sentences to retrieve.

    Returns:
    - List[dict]: List of randomly selected sentences.
    """
    indices = list(range(len(document)))
    indices.remove(query_idx)  # Exclude the target sentence

    # Initialize an empty list to store consistent indices
    selected_indices = []

    # Loop through k values, using seed=i to ensure deterministic selection
    for i in range(1, seed_k + 1):
        random.seed(i + 17 + sentence_idx)  # Set seed for reproducibility
        selected_index = random.sample(indices, 1)[0]  # Select one random index
        selected_indices.append(selected_index)  # Add to the list

    # Retrieve the sentences corresponding to the selected indices
    return [document[i] for i in selected_indices[:k]]



def bertopic_retrieve(document, query_idx, k):
    """
    Retrieve the top-k most similar sentences using BERTopic.

    Args:
    - document (List[dict]): List of sentences in the document.
    - query_idx (int): Index of the target sentence.
    - k (int): Number of sentences to retrieve.

    Returns:
    - List[dict]: List of the top-k retrieved sentences.
    """
    # Cache the BERTopic model and topics for the document
    if not hasattr(bertopic_retrieve, "topic_model"):
        bertopic_retrieve.topic_model = BERTopic()
        bertopic_retrieve.document_topics = {}  # Cache for document topics

    sentences = [sent['value']['text'] for sent in document]

    # Check if this document is already processed
    document_id = id(document)
    if document_id not in bertopic_retrieve.document_topics:
        topics, probs = bertopic_retrieve.topic_model.fit_transform(sentences)
        bertopic_retrieve.document_topics[document_id] = (topics, probs)

    topics, _ = bertopic_retrieve.document_topics[document_id]
    query_topic = topics[query_idx]

    # Retrieve sentences with the same topic as the query
    similar_sentences = [i for i, topic in enumerate(topics) if topic == query_topic and i != query_idx]
    retrieved_sentences = [document[i] for i in similar_sentences[:k]]

    # Print the query and its retrieval results for debugging
    # print("\nQuery Sentence:")
    # print(f" - {document[query_idx]['value']['text']}")
    # print("Retrieved Contexts:")
    # for idx, context in enumerate(retrieved_sentences):
    #     print(f" {idx + 1}: {context['value']['text']}")

    return retrieved_sentences



def write_in_hsln_format(input, output_path, tokenizer, k_min, k_max, context_type="bm25", randomize=False):
    """
    Prepares data in HSLN-compatible format using BM25, Random, Named Entity Overlap, or BERTopic for global context.

    Args:
    - input (List[dict]): Input data (JSON).
    - output_path (str): Path to save the tokenized output.
    - tokenizer: BERT tokenizer instance.
    - k_min (int): Minimum number of context sentences.
    - k_max (int): Maximum number of context sentences.
    - context_type (str): Context retrieval method ('bm25', 'random', 'named_entity_overlap', or 'bertopic').
    - randomize (bool): Whether to randomize the number of sentences retrieved.
    """
    final_string = ""

    for file in tqdm(input, desc="Processing files", unit="file"):
        file_name = file['id']
        annotations = file['annotations'][0]['result']
        sentence_count = len(annotations)

        for sentence_idx, annotation in enumerate(annotations):
            file_id = f"{file_name}/{sentence_idx}"
            final_string += f"###{file_id}\n"

            seed_k = k_max
            # Determine number of context sentences
            k = random.randint(k_min, k_max) if randomize else k_max

            # Retrieve context sentences based on the selected method
            if context_type == "bm25":
                context_sentences = bm25_retrieve(annotations, sentence_idx, k)
            elif context_type == "random":
                context_sentences = random_retrieve(annotations, sentence_idx, k, sentence_idx, seed_k)
            elif context_type == "bertopic":
                context_sentences = bertopic_retrieve(annotations, sentence_idx, k)
            elif context_type == "sentencebert":
                context_sentences = sentencebert_retrieve(annotations, sentence_idx, k)
            elif context_type == "entities":
                context_sentences = common_entities_retrieve(annotations, sentence_idx, k)
            else:
                raise ValueError("Invalid context_type. Choose 'bm25', 'random', 'named_entity_overlap', or 'bertopic'.")

            # Process the target sentence first
            target_sentence = annotation['value']['text']
            target_label = annotation['value']['labels'][0]
            target_sentence = target_sentence.replace("\r", "")
            if target_sentence.strip():
                target_tokens = tokenizer.encode(target_sentence, add_special_tokens=True, max_length=MAX_SEQ_LENGTH)
                target_tokens_txt = " ".join(map(str, target_tokens))
                final_string += f"{target_label}\t{target_tokens_txt}\t1\n"

            # Process the retrieved sentences
            for context_annotation in context_sentences:
                sentence_txt = context_annotation['value']['text']
                sentence_label = context_annotation['value']['labels'][0]
                sentence_txt = sentence_txt.replace("\r", "")

                if sentence_txt.strip():
                    sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True, max_length=MAX_SEQ_LENGTH)
                    sent_tokens_txt = " ".join(map(str, sent_tokens))
                    final_string += f"{sentence_label}\t{sent_tokens_txt}\t0\n"

            final_string += "\n"

    # Write the output
    with open(output_path, "w") as file:
        file.write(final_string)

def tokenize_and_save(input_path, output_dir, file_suffix, tokenizer, k_min, k_max, context_type="bm25", randomize=False):
    """
    Tokenizes the input JSON data and saves it in HSLN format with BM25, Random, Named Entity Overlap, or BERTopic global context.

    Args:
    - input_path (str): Path to the input JSON file.
    - output_dir (str): Directory where the output file will be saved.
    - file_suffix (str): Suffix for the output file name (e.g., 'train_scibert').
    - tokenizer: BERT tokenizer instance.
    - k_min (int): Minimum number of context sentences.
    - k_max (int): Maximum number of context sentences.
    - context_type (str): Context retrieval method ('bm25', 'random', 'named_entity_overlap', or 'bertopic').
    - randomize (bool): Whether to randomize the number of sentences retrieved.
    """
    context_dir = os.path.join(output_dir, context_type)
    os.makedirs(context_dir, exist_ok=True)

    # Load input data
    with open(input_path, "r") as f:
        input_data = json.load(f)

    output_path = os.path.join(context_dir, f"{file_suffix}_scibert.txt")
    write_in_hsln_format(input_data, output_path, tokenizer, k_min, k_max, context_type=context_type, randomize=randomize)
    print(f"Tokenized data saved to: {output_path}")

    # Fixed context sizes for test split
    if "test" in file_suffix:
        for k in range(k_min, k_max + 1):
            fixed_output_path = os.path.join(context_dir, f"test_scibert_{k}.txt")
            write_in_hsln_format(input_data, fixed_output_path, tokenizer, k, k, context_type=context_type, randomize=False)
            print(f"Tokenized data with fixed context size {k} saved to: {fixed_output_path}")

def tokenize():
    """
    Main function to tokenize and process global context using BM25, Random, Named Entity Overlap, or BERTopic.
    """
    if len(sys.argv) != 13:
        print("Usage: python global_context.py --input_dir <input_dir> --output_dir <output_dir> --k_min <k_min> --k_max <k_max> --context_type <bm25|random|named_entity_overlap|bertopic> --randomize <True|False>")
        sys.exit(1)

    input_dir = sys.argv[2]
    output_dir = sys.argv[4]
    k_min = int(sys.argv[6])
    k_max = int(sys.argv[8])
    context_type = sys.argv[10]
    randomize = sys.argv[12].lower() == "true"



    if context_type not in ["bm25", "random", "named_entity_overlap", "bertopic", "sentencebert", "entities"]:
        print("Invalid context_type. Choose 'bm25', 'random', 'named_entity_overlap', or 'bertopic'.")
        sys.exit(1)

    tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)

    # Define input files
    files = {
        "train": os.path.join(input_dir, "train.json"),
        "dev": os.path.join(input_dir, "dev.json"),
        "test": os.path.join(input_dir, "test.json")
    }

    # Process each split
    for split, filepath in files.items():
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        # Randomize for train/dev and fixed retrieval for test
        randomize_split = randomize if split in ["train", "dev"] else False
        tokenize_and_save(filepath, output_dir, split, tokenizer, k_min, k_max, context_type, randomize_split)

if __name__ == "__main__":
    tokenize()
