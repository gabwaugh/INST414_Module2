import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the dataset
tweets_df = pd.read_csv('Tweets.csv')

# Sample a smaller subset for analysis (500 random tweets)
sampled_tweets_df = tweets_df.sample(n=500, random_state=42)

# Filter for negative sentiments only
negative_sentiments = sampled_tweets_df[sampled_tweets_df['airline_sentiment'] == 'negative']

# Create a graph
G = nx.Graph()

# Count negative sentiments per airline
negative_counts = negative_sentiments['airline'].value_counts()

# Add nodes and edges with weights (based on negative sentiment count)
for airline, count in negative_counts.items():
    G.add_node(airline)
    G.add_edge('Negative Sentiment', airline, weight=count)

# Calculate degree centrality (importance of each airline node)
centrality = nx.degree_centrality(G)

# Summarize centrality scores into a dataframe
centrality_df = pd.DataFrame({
    'Airline': list(negative_counts.index),
    'Negative Sentiment Count': negative_counts.values,
    'Degree Centrality': [centrality[airline] for airline in negative_counts.index]
}).sort_values(by='Degree Centrality', ascending=False)

# Print the centrality summary
print(centrality_df)

# Visualize the network graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, k=0.5)

# Draw the graph with node size proportional to centrality
nx.draw_networkx_nodes(G, pos, node_size=[centrality[node] * 5000 for node in G.nodes()], node_color='lightcoral')
nx.draw_networkx_edges(G, pos, width=[d['weight'] / 10 for u, v, d in G.edges(data=True)], alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10)

# Finalize and show the plot
plt.title("Negative Sentiment Network Graph")
plt.axis('off')
plt.show()