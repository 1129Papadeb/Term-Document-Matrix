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
    "Artificial intelligence (AI) is the simulation of human intelligence in machines programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving. AI has become an essential part of the technology industry.",
    "Climate change refers to long-term shifts and alterations in temperature and weather patterns. It can be natural, but recent trends are largely driven by human activities, especially fossil fuel burning. These emissions trap heat, leading to global warming and various ecological impacts.",
    "Quantum computing uses the principles of quantum mechanics to perform calculations at unprecedented speeds. It relies on quantum bits or qubits, which can exist in multiple states simultaneously. This technology has the potential to solve complex problems much faster than classical computers.",
    "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. It generally involves the green pigment chlorophyll and generates oxygen as a byproduct. This process is fundamental to life on Earth.",
    "Blockchain is a distributed ledger technology that allows data to be stored across a network of computers in a secure, transparent way. It underpins cryptocurrencies like Bitcoin and Ethereum. Each block in the chain contains a list of transactions that are cryptographically secured."
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
