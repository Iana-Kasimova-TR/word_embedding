import string

import numpy
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')


def remove_punctuation(text):
    return "".join([i for i in text if i not in string.punctuation])

def replace_underscore_on_whitespace(text):
    replaced = []
    for tag in text:
        replaced.append(tag.replace("_", " "))
    return replaced



