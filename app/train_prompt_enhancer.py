# app/train_prompt_enhancer.py

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load data from CSV
df = pd.read_csv('app/enhancer_data.csv')

# Prepare the training data
X = df['prompt']
y = df['enhanced_prompt']

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# KNN Model for nearest neighbors
model = NearestNeighbors(n_neighbors=1, metric='cosine')
model.fit(X_vec)

# Save the model and vectorizer
joblib.dump(model, 'app/enhancer_model.pkl')
joblib.dump(vectorizer, 'app/enhancer_vectorizer.pkl')

# Save the data backup for reference
df.to_csv('app/_prompt_enhancer_data_backup.csv', index=False)

print("Model and vectorizer have been saved successfully!")
