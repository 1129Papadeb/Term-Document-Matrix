import wikipediaapi
from collections import Counter
import pandas as pd
from math import log, sqrt

# Function to fetch content from Wikipedia using wikipedia-api
def fetch_wikipedia_article(title):
    # Correctly initialize the Wikipedia API with language and user_agent
    wiki = wikipediaapi.Wikipedia(language='en', user_agent="MyApp/1.0 (https://example.com)") 
    page = wiki.page(title)
    return page.text

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

# Fetch Wikipedia articles
titles = ["Artificial_intelligence", "Climate_change", "Quantum_computing", "Photosynthesis", "Blockchain"]
documents = [fetch_wikipedia_article(title) for title in titles]

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
