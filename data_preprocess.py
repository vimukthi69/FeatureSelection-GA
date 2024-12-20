import pandas as pd

# Loading the CSV file
df = pd.read_csv('dataset/Tweets.csv')
df_filtered = df[['text', 'sentiment']]

# Removing rows where sentiment is 'neutral'
df_filtered = df_filtered[df_filtered['sentiment'] != 'neutral']

# Encoding 'positive' to 1 and 'negative' to 2
df_filtered['sentiment'] = df_filtered['sentiment'].map({'positive': 1, 'negative': 0})

# Taking 2500 samples from each sentiment group
df_sampled = df_filtered.groupby('sentiment').apply(lambda x: x.sample(n=1250, random_state=42)).reset_index(drop=True)
print(df_sampled.head())
df_sampled.to_csv('dataset/processed_sentiment_data_0.csv', index=False)
