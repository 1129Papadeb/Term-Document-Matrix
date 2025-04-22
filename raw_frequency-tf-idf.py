from collections import Counter
import pandas as pd
from math import log

# Function to compute raw frequency (without normalization)
def compute_raw_frequency(tokens, vocab):
    count = Counter(tokens)
    return {term: count[term] for term in vocab}

# Function to compute Inverse Document Frequency (IDF)
def compute_idf(tokenized_docs, vocab):
    N = len(tokenized_docs)
    idf_dict = {}
    for term in vocab:
        # Count the number of documents containing the term
        df = sum(term in doc for doc in tokenized_docs)
        # Compute IDF using the formula: idf(t) = log(N / df(t))
        idf_dict[term] = log(N / (df or 1))
    return idf_dict

# Function to compute TF-IDF
def compute_tfidf(tf_vector, idf, vocab):
    return {term: tf_vector[term] * idf[term] for term in vocab}

# Sample corpus (5 documents with different topics)
documents = [
    "World War II was a global war that lasted from 1939 to 1945.",
    "Basketball is a sport played by teams of players who try to shoot the ball into a hoop.",
    "Computer science is the study of computational systems and computers.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "Climate change refers to long-term changes in temperature and weather patterns."
]

# Tokenize and apply lowercase the documents into words
tokenized_docs = [doc.lower().split() for doc in documents]

# Create a set of unique words (vocabulary)
vocabulary = set(word for doc in tokenized_docs for word in doc)

# Convert vocabulary set to a list to use it as DataFrame columns
vocabulary_list = list(vocabulary)

# Compute the raw frequency for each document (without normalization)
raw_frequency_vectors = [compute_raw_frequency(doc, vocabulary) for doc in tokenized_docs]

# Create a DataFrame for the Raw Frequency Term-Document Matrix
raw_frequency_matrix = pd.DataFrame(raw_frequency_vectors, columns=vocabulary_list).fillna(0)

# Compute the Inverse Document Frequency (IDF)
idf = compute_idf(tokenized_docs, vocabulary)

# Compute the TF-IDF vectors
tfidf_vectors = [compute_tfidf(tf, idf, vocabulary) for tf in raw_frequency_vectors]

# Create a DataFrame for the TF-IDF Term-Document Matrix
tfidf_matrix = pd.DataFrame(tfidf_vectors, columns=vocabulary_list).fillna(0)

# Display the results
print("Term-Document Matrix (Raw Frequency):")
print(raw_frequency_matrix)
print("\nTerm-Document Matrix (TF-IDF Weights):")
print(tfidf_matrix)
