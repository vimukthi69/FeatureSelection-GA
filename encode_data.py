from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('dataset/processed_sentiment_data_2.csv')

# Load the all-MiniLM-L6-v2 model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Encode the text column
X = model.encode(df['text'].tolist(), show_progress_bar=True)

# Display the shape of the encoded features
print("Shape of encoded data:", X.shape)

# Save the encoded data if needed
np.save('encoded_text_2.npy', X)  # Save to a .npy file for future use
