from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

# Sample corpus (5 documents)
documents = [
    "Artificial intelligence (AI) is the simulation of human intelligence in machines programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving. AI has become an essential part of the technology industry.",
    "Climate change refers to long-term shifts and alterations in temperature and weather patterns. It can be natural, but recent trends are largely driven by human activities, especially fossil fuel burning. These emissions trap heat, leading to global warming and various ecological impacts.",
    "Quantum computing uses the principles of quantum mechanics to perform calculations at unprecedented speeds. It relies on quantum bits or qubits, which can exist in multiple states simultaneously. This technology has the potential to solve complex problems much faster than classical computers.",
    "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. It generally involves the green pigment chlorophyll and generates oxygen as a byproduct. This process is fundamental to life on Earth.",
    "Blockchain is a distributed ledger technology that allows data to be stored across a network of computers in a secure, transparent way. It underpins cryptocurrencies like Bitcoin and Ethereum. Each block in the chain contains a list of transactions that are cryptographically secured."
]

# Corresponding labels for classification
labels = [0, 1, 2, 3, 4]  # Each doc is from a different category

# Tokenize the documents
tokenized_docs = [doc.lower().split() for doc in documents]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

# Create dense document vectors by averaging word embeddings
def get_doc_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

doc_vectors = np.array([get_doc_vector(doc, model) for doc in tokenized_docs])

# Train Logistic Regression
classifier = LogisticRegression(max_iter=1000)
classifier.fit(doc_vectors, labels)

# Predict on the same data (not recommended in real scenarios)
predictions = classifier.predict(doc_vectors)

# Output the classification report
print("Classification Report:\n")
print(classification_report(labels, predictions, zero_division=1))
