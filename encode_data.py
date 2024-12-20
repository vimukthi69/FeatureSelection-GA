from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Loading the CSV file
df = pd.read_csv('dataset/processed_sentiment_data_0.csv')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Encoding the text column
X = model.encode(df['text'].tolist(), show_progress_bar=True)
print("Shape of encoded data:", X.shape)
np.save('encoded_text_0.npy', X)
