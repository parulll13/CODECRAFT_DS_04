import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

df = pd.read_csv('twitter_training.csv', header=None)
df.columns = ['ID', 'Entity', 'Sentiment', 'Tweet']
df.head()
print("Dataset Shape:", df.shape)
print("Missing Values:\n", df.isnull().sum())
print("Sentiment Counts:\n", df['Sentiment'].value_counts())

#Sentiment Distribution Visualization
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Sentiment', hue='Sentiment', palette='Set2', legend=False)
plt.title('Distribution of Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Tweet Count')
plt.show()

#Sentiment by Entity
top_entities = df['Entity'].value_counts().head(10).index
top_df = df[df['Entity'].isin(top_entities)]
plt.figure(figsize=(10, 6))
sns.countplot(data=top_df, y='Entity', hue='Sentiment', palette='Set1')
plt.title('Sentiment Distribution by Top 10 Entities')
plt.xlabel('Tweet Count')
plt.ylabel('Entity')
plt.legend(title='Sentiment')
plt.show()

#WordCloud for Positive, Negative, Neutral
def plot_wordcloud(data, sentiment):
    text_series = data[data['Sentiment'] == sentiment]['Tweet'].dropna()
    text = ' '.join(text_series.astype(str))  # Ensure everything is string
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment} Sentiment')
    plt.show()

for sentiment in ['Positive', 'Negative', 'Neutral']:
    plot_wordcloud(df, sentiment)

#Entity-wise Sentiment Ratio
entity = 'Apple'  # change to any other entity you like
entity_df = df[df['Entity'] == entity]

plt.figure(figsize=(5,5))
entity_df['Sentiment'].value_counts().plot.pie(autopct='%1.1f%%', colors=['green', 'red', 'grey'])
plt.title(f'Sentiment Distribution for {entity}')
plt.ylabel('')
plt.show()
