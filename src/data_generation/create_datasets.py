import os

from data_generation.manipulate_dataset import manipulate_dataset
from helper.utils import preprocess_batch
from helper.parse_args import parse_args

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_CLEAN = os.path.join(BASE_DIR, "..", "..", "data", "clean")
DATA_PATH_MANIPULATED = os.path.join(
    BASE_DIR, "..", "..", "data", "manipulated")
ARGS = parse_args()


def get_clean_set(dataset, set_size):
    clean_dataset = generate_subset(dataset, set_size)

    prefix = ARGS.model.replace("/", "_")
    file_name = f'{prefix}_{set_size}.jsonl'
    final_path = os.path.join(DATA_PATH_CLEAN, file_name)
    clean_dataset.to_json(final_path)
    return clean_dataset


def get_manipulated_set(clean_dataset, model, tokenizer, method, poisoning_rate, bit_sequence):
    print("Creating manipulated dataset from Base Dataset...")
    manipulated_dataset = manipulate_dataset(
        clean_dataset, poisoning_rate, bit_sequence, model, tokenizer, method)

    prefix = ARGS.model.replace("/", "_")
    file_name = f'{prefix}_{len(clean_dataset)}_{bit_sequence}_manipulated.jsonl'
    final_path = os.path.join(DATA_PATH_MANIPULATED, method, file_name)
    manipulated_dataset.to_json(final_path)
    print("Successful creation of manipulated dataset")
    return manipulated_dataset


def get_train_test_splits(dataset, tokenizer, seed=42):
    dataset_dict = dataset.train_test_split(test_size=0.3, seed=seed)
    tokenized_dataset_train = dataset_dict["train"]
    tokenized_dataset_test = dataset_dict["test"]

    # DONT REMOVE remove_columns!!!!!!!!!!!!!
    train_set = tokenized_dataset_train.map(lambda batch: preprocess_batch(batch, tokenizer), batched=True,
                                            remove_columns=tokenized_dataset_train.column_names)
    test_set = tokenized_dataset_test.map(lambda batch: preprocess_batch(batch, tokenizer), batched=True,
                                          remove_columns=tokenized_dataset_test.column_names)

    return train_set, test_set


def generate_subset(dataset, size):
    """
    Creates a subset from a base dataset with dynamic sizes
    :param dataset: Dataset where the subset is generated from
    :param size: Size of the wanted subset
    """
    # get training data_generation, since instructions dataset only has training
    dataset = dataset['train']
    if dataset.num_rows < int(size):
        raise ValueError("size parameter too large")
    return dataset.select(range(size))
