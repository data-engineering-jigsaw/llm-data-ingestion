import nltk

nltk.download('punkt')  # Download necessary datasets
from nltk.tokenize import word_tokenize

text = "I am sleepwalking, are you?"
tokens = word_tokenize(text)
print(tokens)