from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import os

# Function to load article content from local file
def load_article_from_file(title):
    filepath = f"wiki_articles/{title}.txt"
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# List of Wikipedia article titles (also used as filenames)
titles = [
    "Artificial_intelligence",
    "Climate_change",
    "Quantum_computing",
    "Photosynthesis",
    "Blockchain"
]

# Load articles from local folder
documents = [load_article_from_file(title) for title in titles]

# Corresponding labels for classification
labels = [0, 1, 2, 3, 4]  # Each document has a unique label

# Tokenize the documents (lowercase and split on whitespace)
tokenized_docs = [doc.lower().split() for doc in documents]

# Train the Word2Vec model
model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

# Function to get average word vector for a document
def get_doc_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Create document vectors by averaging word vectors
doc_vectors = np.array([get_doc_vector(doc, model) for doc in tokenized_docs])

# Train a Logistic Regression model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(doc_vectors, labels)

# Predict the same labels (just for checking)
predictions = classifier.predict(doc_vectors)

# Output the classification report
print("Classification Report:\n")
print(classification_report(labels, predictions, zero_division=1))
