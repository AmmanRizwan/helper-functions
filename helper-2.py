"""
Focusing Mainly in unSupervised Learning Datasets

Helper Functions for Data Analysis Visualization

This module provides a collection of utility functions for data analysis particularly 
"""

# Pair Plots

def visualize_pair_plot(df):
  pass


# Dimensionality Reduction

def visualize_dimesion_reduction(df):
  pass

# Clusters Ploting

def visualize_clusters():
  plt.scatter(df["Feature 1"], df['Feature 2'], c=df['Cluster'], cmap='viridis', s=50)
  plt.xlabel("Feature 1")
  plt.ylabel("Feature 2")
  plt.title("Clusters Found by K-Means")
  plt.show()

# KMeans prediction

def kmean_prediction():
  score = silhouette_score(df[['Feature', "Features..."]], df['Cluster'])
  print("Score:", score)
  
# DBSCAN prediction

def dbscan_prediction():
  mask = cluster_labels != -1
  score = silhouette_score(X[mask], cluster_labels[mask])
  print("Score", score)
  
def remove_html(text):
  pattern = re.compile('<.*?>')
  return pattern.sub('', text)

def remove_url(text):
  pattern = re.compile(r'https?://\S+|www\.\S+')
  return pattern.sub("", text)

def remove_mentions(text):
  pattern = re.compile("@\S+")
  return pattern.sub("", text)

def remove_digits(text):
  pattern = re.compile(r'\d+')
  return pattern.sub("", text)

def remove_symbols(text):
  return re.sub(r'[^\x00-\x7F]+', "", str(text))


def remove_punc(text):
  punc = string.punctuation
  text_nonPunc = ''.join([char for char in text if char not in punc])
  return text_nonPunc