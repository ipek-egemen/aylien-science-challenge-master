from transformers import CanineForSequenceClassification
from transformers import CanineTokenizer

# our sentiment classifier is based on Google's Canine-S
# Canine-S was fine-tuned with our data and performs quite good
# Performance: 
# Accuracy: 0.9159647404505387
# Precision: 0.915980373907115
# Recall: 0.9163266794115629
# F1 Macro: 0.9158494095465145
# F1 Micro: 0.9159647404505387
# More information about the pre-trained Canine-S: https://huggingface.co/google/canine-s

# We are using transformers library to initialize, train, save and load our model
# More info: https://huggingface.co/docs/transformers/index

# load the model
def load_sentiment_classifier(path):
    model = CanineForSequenceClassification.from_pretrained(path)
    return model
# load the tokenizer for the model
def load_tokenizer(path):
    tokenizer = CanineTokenizer.from_pretrained(path)
    return tokenizer