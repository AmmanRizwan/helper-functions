"""
Focusing Mainly in unSupervised Learning Datasets

Helper Functions for Data Analysis Visualization

This module provides a collection of utility functions for data analysis particularly 
"""

# Pair Plots

def visualize_pair_plot(df):
  plt.figure(figsize=(10, 6))
  sns.pairplot(dataset, hue='cluter', palette='viridis', diag_king="kde")
  plt.suptitle("Pair Plot of Clusters")
  plt.show()


# Dimensionality Reduction

def visualize_dimesion_reduction(df):
  # Need the PCA Transform before ploting
  plt.figure(10, 6)
  plt.scatter(df_pca['PCA1'], df_pca["PCA2"], c=df_pca['cluter'])
  plt.title("Clusters after PCA Dimension Reduction")
  plt.xlabel("Principle Component 1")
  plt.ylabel("Principle Component 2")
  plt.show()

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

def drop_column(df, columns):
  df.drop(columns=[columns], inplace=True)
  
def fnn_model():
  model = keras.Sequential([
    tf.keras.layers.Input(shape=(df.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
  
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  
  model.summary()
  
def cnn_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
  
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  
  model.summary()
  
def rnn_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(df.shape[1],))
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(1, activiation='sigmoid'),
  ])
  
  model.compile(optimizer='adam', loss='crossentropy', metrics=['accuracy'])
  
  model.summary()