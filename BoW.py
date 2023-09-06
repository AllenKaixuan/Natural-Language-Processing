from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

vectorizer = TfidfVectorizer()

# transform the documents into BoW representation

X = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

# use TF-IDF weights

for i, doc in enumerate(documents):
    print(f"Document {i+1}:")
    for j, word in enumerate(feature_names):
        weight = X[i, j]
        if weight > 0:
            print(f"   {word}: {weight:.4f}")