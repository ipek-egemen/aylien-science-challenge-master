# external
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import re
import spacy
import typing
from typing import List, Tuple, Dict
from transformers import CanineTokenizer, CanineForSequenceClassification
from datasets import Dataset, DatasetDict

# imports from modules
from in_out import read_txt, out_path, write_txt
from load_model_tokenizer import load_sentiment_classifier, load_tokenizer
from pre_processing import remove_numbers, remove_handle_content, remove_url_content, remove_stopwords_punctuation, remove_entities, lemmatize_documents
from make_dataset import to_dataset, tokenize_dataset, make_dataloader
from predict import predict_sentiment, model_pred_to_label
import os

# ask for input path
current_dir = os.getcwd()


path = input('Directory to the input path')
# there is a function out_path when used
# generates an output.txt in the same directory 

# ask for output path
export_path = input('Directory of the export path')

# load model and tokenizer

sentiment_classifier = load_sentiment_classifier(f'{current_dir}/sentiment_classifier/model/sentiment_classifier')
tokenizer = load_tokenizer(f'{current_dir}/sentiment_classifier/model/tokenizer')
def tokenizer_fn(example: str):
    return tokenizer(example['text'], padding='max_length', truncation=True)

# read .txt file and get documents
documents = read_txt(path)

# pre-process documents
nlp = spacy.load('en_core_web_sm')
documents = lemmatize_documents(
    remove_entities(
        remove_stopwords_punctuation(
            remove_url_content(
                remove_handle_content(
                    remove_numbers(documents))))))

# make a dataset from documents
dataset = to_dataset(documents)

#tokenize documents
dataset_tokenized = tokenize_dataset(tokenizer_fn, dataset)

# load tokenized dataset into a dataloader
dataset_dataloader = make_dataloader(dataset_tokenized)

# get predictions
predictions = predict_sentiment(sentiment_classifier, dataset_dataloader)

# convert predictions into labels of 'positive', 'neutral', and 'negative
prediction_labels = model_pred_to_label(predictions)

# export predictions to an output.txt file in the same directory as the input file
write_txt(export_path, prediction_labels)

