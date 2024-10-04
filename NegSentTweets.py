import pandas as pd  
import networkx as nx  
import matplotlib.pyplot as plt  

# Load the dataset
tweets_df = pd.read_csv('Tweets.csv')

# Sample a smaller subset for analysis
sampled_tweets_df = tweets_df.sample(n=500, random_state=42)

# Filter for negative sentiments
negative_sentiments = sampled_tweets_df[sampled_tweets_df['airline_sentiment'] == 'negative']

# Create a graph
G = nx.Graph()

# Count negative sentiments per airline
negative_counts = negative_sentiments['airline'].value_counts()

# Add nodes and edges to the graph
for airline, count in negative_counts.items():
    G.add_node(airline)  
    G.add_edge('Negative Sentiment', airline, weight=count)  

# Set up the plot
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, k=0.5)

# Draw the graph
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightcoral')
nx.draw_networkx_edges(G, pos, width=[d['weight'] for u, v, d in G.edges(data=True)], alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10)

# Finalize and show the plot
plt.title("Negative Sentiment Frequency towards Airlines (Sampled Data)")
plt.axis('off')
plt.show()