from typing import List
import typing
from datasets import Dataset
from torch.utils.data import DataLoader

# this module uses the HuggingFace Datasets library which uses Apache Arrow for data storage
# More info: https://huggingface.co/docs/datasets/index
# we also load the data into the model with Pytorch DataLoader object
# More info: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

# takes in a list of documents/examples created with the read_txt function in the in_out module.
# first, convert the list into a dictionary by setting a key 'text' and value to the documents/examples created
# then use that dictionary to create a dataset object and return it
def to_dataset(documents: List[str]) -> Dataset:
    dataset = {'text': documents}
    dataset = Dataset.from_dict(dataset)
    return dataset

# apply tokenize function to the dataset and get tokenized dataset
def tokenize_dataset(tokenizer_fn, dataset: Dataset) -> Dataset:
    dataset_tokenized = dataset.map(tokenizer_fn)
    dataset_tokenized = dataset_tokenized.remove_columns(['text']) #'text' column is no longer needed, if kept, it will cause the model to crash
    dataset_tokenized.set_format('torch') #setting tensor format to torch
    return dataset_tokenized

# make a dataloader for the dataset
def make_dataloader(dataset_tokenized: Dataset) -> DataLoader:
    dataset_dataloader = DataLoader(dataset_tokenized, batch_size=4)
    return dataset_dataloader                                      


