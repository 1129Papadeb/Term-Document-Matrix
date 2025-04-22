from collections import Counter
import pandas as pd
from math import log
import os

# Function to load Wikipedia article text from a local file
def load_article_from_file(title):
    filepath = f"wiki_articles/{title}.txt"
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# Function to compute raw frequency (without normalization)
def compute_raw_frequency(tokens, vocab):
    count = Counter(tokens)
    return {term: count.get(term, 0) for term in vocab}

# Function to compute Inverse Document Frequency (IDF)
def compute_idf(tokenized_docs, vocab):
    N = len(tokenized_docs)
    idf_dict = {}
    for term in vocab:
        df = sum(term in doc for doc in tokenized_docs)
        idf_dict[term] = log(N / (df or 1))  # Avoid division by zero
    return idf_dict

# Function to compute TF-IDF
def compute_tfidf(tf_vector, idf, vocab):
    return {term: tf_vector[term] * idf[term] for term in vocab}

# List of Wikipedia article titles (same as filenames without .txt)
titles = [
    "Artificial_intelligence",
    "Climate_change",
    "Quantum_computing",
    "Photosynthesis",
    "Blockchain"
]

# Load articles from local text files
documents = [load_article_from_file(title) for title in titles]

# Tokenize and lowercase
tokenized_docs = [doc.lower().split() for doc in documents]

# Build vocabulary
vocabulary = set(word for doc in tokenized_docs for word in doc)
vocabulary_list = list(vocabulary)

# Compute raw term frequencies for each document
raw_frequency_vectors = [compute_raw_frequency(doc, vocabulary) for doc in tokenized_docs]

# Create Term-Document Matrix (Raw Frequency)
raw_frequency_matrix = pd.DataFrame(raw_frequency_vectors, columns=vocabulary_list).fillna(0)

# Compute IDF values
idf = compute_idf(tokenized_docs, vocabulary)

# Compute TF-IDF vectors
tfidf_vectors = [compute_tfidf(tf, idf, vocabulary) for tf in raw_frequency_vectors]

# Create Term-Document Matrix (TF-IDF)
tfidf_matrix = pd.DataFrame(tfidf_vectors, columns=vocabulary_list).fillna(0)

# Display results
print("Term-Document Matrix (Raw Frequency):")
print(raw_frequency_matrix)
print("\nTerm-Document Matrix (TF-IDF Weights):")
print(tfidf_matrix)
