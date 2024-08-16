from typing import List
import typing
import re
import spacy

# this module uses spacy
# https://spacy.io/usage/spacy-101

nlp = spacy.load('en_core_web_sm')
# this function removes numbers from a list of strings
def remove_numbers(data: List[str]) -> List[str]:
    clean_data = []
    pattern = r'\d+'
    for document in data:
        document = re.sub(pattern, '', document)
        clean_data.append(document)
    return clean_data

# this function removes handle content from documents example: @someuser
def remove_handle_content(data: List[str]) -> List[str]:
    clean_data = []
    for document in data:
        document = ' '.join([token for token in document.split() if '@' not in token])
        clean_data.append(document)
    return clean_data

# this function removes url content from documents example: http//:someurl
def remove_url_content(data: List[str]) -> List[str]:
    clean_data = []
    for document in data:
        document = ' '.join([token for token in document.split() if token[:4]!='http'])
        clean_data.append(document)
    return clean_data

# this function removes stopwords and punctuation from documents example: the !, etc.
def remove_stopwords_punctuation(data: List[str]) -> List[str]:
    clean_data = []
    for document in data:
        doc = nlp(document)
        document = ' '.join([token.text for token in doc if not token.is_stop and not token.is_punct])
        clean_data.append(document)
    return clean_data

# this function removed named entities form the example
def remove_entities(data: List[str]) -> List[str]:
    clean_data = []
    for document in data:
        doc = nlp(document)
        document = ' '.join([token.text for token in doc if not token.ent_type_])
        clean_data.append(document)
    return clean_data

# this function lemmatizes tokens in the documents inside the data example: going -> go # this also lowercases letters
# this function also removes duplicate tokens
def lemmatize_documents(data: List[str]) -> List[str]:
    clean_data = []
    for document in data:
        doc = nlp(document)
        document = ' '.join([token.lemma_.lower() for token in doc])
        clean_data.append(document)
    return clean_data