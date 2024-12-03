import pandas as pd

# Load the CSV file
df = pd.read_csv('dataset/Tweets.csv')

# Select only the 'text' and 'sentiment' columns
df_filtered = df[['text', 'sentiment']]

# Remove rows where sentiment is 'neutral'
df_filtered = df_filtered[df_filtered['sentiment'] != 'neutral']

# Map 'positive' to 1 and 'negative' to 2
df_filtered['sentiment'] = df_filtered['sentiment'].map({'positive': 1, 'negative': 0})

# Take 2500 samples from each sentiment group
df_sampled = df_filtered.groupby('sentiment').apply(lambda x: x.sample(n=500, random_state=42)).reset_index(drop=True)

# Display the resulting dataframe
print(df_sampled.head())

# Save the sampled dataframe to a new CSV if needed
df_sampled.to_csv('dataset/processed_sentiment_data_2.csv', index=False)
