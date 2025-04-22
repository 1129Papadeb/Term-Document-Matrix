from collections import Counter
import pandas as pd
from math import log, sqrt

# Function to compute raw frequency (without normalization)
def compute_raw_frequency(tokens, vocab):
    count = Counter(tokens)
    return {term: count[term] for term in vocab}

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

# Sample documents
documents = [
    "World War II was a global war that lasted from 1939 to 1945.",
    "Basketball is a sport played by teams of players who try to shoot the ball into a hoop.",
    "Computer science is the study of computational systems and computers.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "Climate change refers to long-term changes in temperature and weather patterns."
]

# Tokenize documents
tokenized_docs = [doc.lower().split() for doc in documents]

# Vocabulary
vocabulary = set(word for doc in tokenized_docs for word in doc)
vocab_list = list(vocabulary)

# Raw frequency and TF-IDF
raw_freqs = [compute_raw_frequency(doc, vocabulary) for doc in tokenized_docs]
idf = compute_idf(tokenized_docs, vocabulary)
tfidf_vectors = [compute_tfidf(tf, idf, vocabulary) for tf in raw_freqs]

# Compare all pairs using cosine similarity
max_score = 0
most_similar = (0, 0)
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        sim = cosine_similarity(tfidf_vectors[i], tfidf_vectors[j])
        print(f"Similarity between Document {i} and Document {j}: {sim:.4f}")
        if sim > max_score:
            max_score = sim
            most_similar = (i, j)

# Output the most similar documents
print(f"\nThe most similar documents are Document {most_similar[0]} and Document {most_similar[1]} with a cosine similarity of {max_score:.4f}")
