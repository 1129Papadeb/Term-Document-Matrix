from collections import Counter
import pandas as pd
from math import log, sqrt
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
        idf_dict[term] = log(N / (df or 1))
    return idf_dict

# Function to compute TF-IDF
def compute_tfidf(tf_vector, idf, vocab):
    return {term: tf_vector[term] * idf[term] for term in vocab}

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    dot = sum(vec1[t] * vec2[t] for t in vec1)
    mag1 = sqrt(sum(vec1[t]**2 for t in vec1))
    mag2 = sqrt(sum(vec2[t]**2 for t in vec2))
    return dot / (mag1 * mag2) if mag1 != 0 and mag2 != 0 else 0

# Wikipedia article titles (matching file names)
titles = [
    "Artificial_intelligence",
    "Climate_change",
    "Quantum_computing",
    "Photosynthesis",
    "Blockchain"
]

# Load documents from local folder
documents = [load_article_from_file(title) for title in titles]

# Tokenize and lowercase
tokenized_docs = [doc.lower().split() for doc in documents]

# Build vocabulary
vocabulary = set(word for doc in tokenized_docs for word in doc)
vocab_list = list(vocabulary)

# Compute raw frequency
raw_freqs = [compute_raw_frequency(doc, vocabulary) for doc in tokenized_docs]

# Compute IDF and TF-IDF
idf = compute_idf(tokenized_docs, vocabulary)
tfidf_vectors = [compute_tfidf(tf, idf, vocabulary) for tf in raw_freqs]

# Cosine similarity comparison
max_score = 0
most_similar = (0, 0)
print("Pairwise Cosine Similarities:\n")
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        sim = cosine_similarity(tfidf_vectors[i], tfidf_vectors[j])
        print(f"Similarity between '{titles[i]}' and '{titles[j]}': {sim:.4f}")
        if sim > max_score:
            max_score = sim
            most_similar = (i, j)

# Print most similar pair
print(f"\n🔍 The most similar documents are '{titles[most_similar[0]]}' and '{titles[most_similar[1]]}' with a cosine similarity of {max_score:.4f}")
