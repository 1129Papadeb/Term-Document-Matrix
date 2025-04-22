import wikipediaapi
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

# Function to fetch content from Wikipedia using wikipedia-api
def fetch_wikipedia_article(title):
    # Initialize the Wikipedia API with language and user agent
    wiki = wikipediaapi.Wikipedia(language='en', user_agent="MyApp/1.0 (https://example.com)") 
    page = wiki.page(title)
    return page.text  # Return the text content of the page

# List of Wikipedia article titles to fetch
titles = [
    "Artificial_intelligence",
    "Climate_change",
    "Quantum_computing",
    "Photosynthesis",
    "Blockchain"
]

# Fetch articles from Wikipedia
documents = [fetch_wikipedia_article(title) for title in titles]

# Remove any documents that failed to fetch
documents = [doc for doc in documents if doc is not None]

# Corresponding labels for classification (same as before)
labels = [0, 1, 2, 3, 4]  # Each doc is from a different category

# Tokenize the documents (splitting by whitespace and making all words lowercase)
tokenized_docs = [doc.lower().split() for doc in documents]

# Train the Word2Vec model using the tokenized documents
model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

# Function to get the average word vector for a document
def get_doc_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Create document vectors by averaging the word embeddings
doc_vectors = np.array([get_doc_vector(doc, model) for doc in tokenized_docs])

# Train a Logistic Regression model on the document vectors
classifier = LogisticRegression(max_iter=1000)
classifier.fit(doc_vectors, labels)

# Predict the labels for the same data (not recommended for real-world evaluation)
predictions = classifier.predict(doc_vectors)

# Output the classification report
print("Classification Report:\n")
print(classification_report(labels, predictions, zero_division=1))
