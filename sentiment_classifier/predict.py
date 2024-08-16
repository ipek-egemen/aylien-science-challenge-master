from transformers import CanineForSequenceClassification
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import typing
from typing import List

# we load the data into the model with Pytorch DataLoader object
# More info: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# we are using softmax to turn our logits into probabilities
# More info: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
# we are selecting the class with the highest probabiltity using argmax
# More info: https://pytorch.org/docs/stable/generated/torch.argmax.html

# uses the sentiment_classifier to predict examples inside the dataset
def predict_sentiment(sentiment_classifier: CanineForSequenceClassification, dataset_dataloader: DataLoader) -> List[int]:
    predictions = [] # collec model predictions here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set the device to cuda
    sentiment_classifier.to(device) # move the model to the device
    softmax = nn.Softmax(dim=1) # initiate softmax to convert logits into probabilities
    for batch in dataset_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()} #unpack dataloader batch into a dict
            with torch.no_grad(): 
                outputs = sentiment_classifier(**batch)
                logits = outputs.logits
                batch_probs = softmax(logits) # apply softmax to logits and get probabilities
                batch_preds = torch.argmax(batch_probs, dim=1) # select the most probable class
            for pred in batch_preds:
                predictions.append(int(pred.item())) # add predictions for this batch to the predictions list
    return predictions

# this function converts model predictions, which are in the form of integers
# to human readable text
# for the model the values for the labels positive, neutral, negative
# was set to 0, 1, 2 respectively
def model_pred_to_label(predictions: List[int]) -> List[str]:
    prediction_labels = []
    for prediction in predictions:
        if prediction == 0:
             prediction_labels.append('positive')
        elif prediction == 1:
             prediction_labels.append('neutral')
        else:
             prediction_labels.append('negative')
    return prediction_labels
          
    

