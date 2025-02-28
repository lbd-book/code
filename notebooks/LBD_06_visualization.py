#!/usr/bin/env python
# coding: utf-8

# # 6. Visualization Module
# 
# This module provides visualization functions to help users better understand the results.
# 

# In[1]:


from typing import List, Optional
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
import seaborn as sns

import plotly.graph_objects as go
from sklearn.decomposition import PCA
import random


# This function plot_wordcloud takes an input text string, an optional integer specifying the maximum number 
# of words to display in the word cloud (default: 100), and an optional title for the word cloud plot (default: None). 
# The function creates a word cloud from the input text and plots it using the wordcloud and matplotlib libraries.
# 
# The function first creates a WordCloud object with the specified maximum number of words, a white background color, 
# and the 'viridis' colormap. The function then generates the word cloud from the input text using the generate() method 
# of the WordCloud object.
# 
# The function plots the word cloud using the imshow() function from the matplotlib.pyplot module, with bilinear interpolation 
# to smooth the image. The function hides the x and y axes using the axis('off') command.
# 
# The function sets the title of the plot if the title parameter is provided, using the title() function from 
# the matplotlib.pyplot module with a font size of 16 and a bold font weight. 
# Finally, the function displays the plot using the show() function from the matplotlib.pyplot module.
# 
# 

# In[2]:


def plot_wordcloud(text: str, max_words: int = 100, title: Optional[str] = None):
    """
    Plot a word cloud from the input text.
    :param text: str, the input text for the word cloud
    :param max_words: int, the maximum number of words to display in the word cloud (default: 100)
    :param title: Optional[str], an optional title for the word cloud plot (default: None)
    """
    # Create a WordCloud object with the specified maximum number of words
    wordcloud = WordCloud(max_words=max_words, background_color='white', colormap='viridis')

    # Generate the word cloud from the input text
    wordcloud.generate(text)

    # Plot the word cloud
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Set the title if provided
    if title:
        plt.title(title, fontsize=16, fontweight='bold')

    # Display the plot
    plt.show()



# This function plot_sentiment_histogram takes a list of sentiment scores, an optional integer specifying 
# the number of bins to use in the histogram (default: 20), and an optional title for the histogram plot (default: None). 
# The function creates a histogram of the sentiment scores using the matplotlib library.
# 
# The function first creates a new plot with a specified figure size using the figure() function from the matplotlib.pyplot module. 
# Then, it plots the histogram of sentiment scores using the hist() function from the matplotlib.pyplot module, 
# with the specified number of bins, a blue color, an alpha value of 0.7 for transparency, and a black edge color.
# 
# The function sets the title of the plot if the title parameter is provided, using the title() function 
# from the matplotlib.pyplot module with a font size of 16 and a bold font weight. 
# The function sets the x and y labels using the xlabel() and ylabel() functions from the matplotlib.pyplot module with font sizes of 14.
# 
# Finally, the function displays the plot using the show() function from the matplotlib.pyplot module.
# 
# 

# In[3]:


def plot_sentiment_histogram(sentiment_scores: List[float], bins: int = 20, title: Optional[str] = None):
    """
    Plot a histogram of sentiment scores.
    :param sentiment_scores: List[float], a list of sentiment scores
    :param bins: int, the number of bins to use in the histogram (default: 20)
    :param title: Optional[str], an optional title for the histogram plot (default: None)
    """
    # Plot the histogram of sentiment scores
    plt.figure(figsize=(10, 6))
    plt.hist(sentiment_scores, bins=bins, color='blue', alpha=0.7, edgecolor='black')

    # Set the title if provided
    if title:
        plt.title(title, fontsize=16, fontweight='bold')

    # Set x and y labels
    plt.xlabel('Sentiment Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    # Display the plot
    plt.show()



# This function plot_clusters_2d takes a document-term matrix as a CSR matrix, a clustering model used for the documents, 
# and an optional title for the cluster plot (default: None). The function creates a 2D representation of clusters using 
# the first two principal components and plots it using the matplotlib library.
# 
# The function first performs dimensionality reduction on the input matrix using the TruncatedSVD class 
# from the sklearn.decomposition module with two components. Then, it standardizes the reduced matrix 
# using the StandardScaler class from the sklearn.preprocessing module.
# 
# The function gets the cluster labels from the clustering model using its labels_ attribute. 
# Then, it creates a scatter plot of the reduced matrix with different colors for each cluster 
# using the scatter() function from the matplotlib.pyplot module. 
# The scatter plot is created separately for each unique cluster label.
# 
# The function sets the title of the plot if the title parameter is provided, 
# using the title() function from the matplotlib.pyplot module with a font size of 16 and a bold font weight. 
# The function sets the x and y labels using the xlabel() and ylabel() functions from the matplotlib.pyplot module with font sizes of 14.
# 
# The function adds a legend to the plot using the legend() function from the matplotlib.pyplot module. 
# Finally, the function displays the plot using the show() function from the matplotlib.pyplot module.
# 
# 

# In[4]:


def plot_clusters_2d(matrix: csr_matrix, clustering_model: BaseEstimator, title: Optional[str] = None):
    """
    Plot a 2D representation of clusters using the first two principal components.
    :param matrix: csr_matrix, the document-term matrix
    :param clustering_model: BaseEstimator, the clustering model used for the documents
    :param title: Optional[str], an optional title for the cluster plot (default: None)
    """
    # Perform dimensionality reduction using TruncatedSVD
    svd = TruncatedSVD(n_components=2, random_state=42)
    reduced_matrix = svd.fit_transform(matrix)

    # Standardize the reduced matrix
    scaler = StandardScaler()
    standardized_matrix = scaler.fit_transform(reduced_matrix)

    # Get the cluster labels from the clustering model
    cluster_labels = clustering_model.labels_

    # Create a scatter plot of the reduced matrix with different colors for each cluster
    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        plt.scatter(standardized_matrix[cluster_labels == label, 0],
                    standardized_matrix[cluster_labels == label, 1],
                    label=f'Cluster {label}',
                    alpha=0.7)

    # Set the title if provided
    if title:
        plt.title(title, fontsize=16, fontweight='bold')

    # Set x and y labels
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()


# Plot *bow* and *tfidf* matrix.

# In[5]:


def plot_bow_tfidf_matrix(title: str, any_matrix: np.ndarray, ids: List, words: List, as_int = True):
    """
    Plot a 2D representation of clusters using the first two principal components.
    :param any_matrix: csr_matrix, the document-term matrix
    :param clustering_model: BaseEstimator, the clustering model used for the documents
    :param title: Optional[str], an optional title for the cluster plot
    """
 
    plt.figure(figsize=(15, 9))
    # plt.rcParams.update({'font.size': 22})
    
    # Plotting the heatmap
    if as_int:
        fmt_str = '.0f'
    else:
        fmt_str = '.2f'
    sns.heatmap(any_matrix, annot=True, fmt=fmt_str, cmap='Greens', \
                xticklabels=words, yticklabels=[f"{ids[i]}" for i in range(any_matrix.shape[0])])
    
    plt.title(title + " Matrix Visualization")
    plt.xlabel("Words")
    plt.ylabel("Documents")
    plt.show()
    # 'coolwarm'
    


# The function `visualize_tfidf_pca_interactive` creates an interactive PCA plot of TF-IDF data. This visualization is useful for exploring document relationships and identifying patterns.
# 
# **Functionality**
# 
# 1. *TF-IDF Matrix Preparation*:
#    The script begins by preparing the TF-IDF matrix:
#    ```python
#    tfidf_matrix_transposed = np.squeeze(np.asarray(tfidf_matrix))
#    if transpose:
#        tfidf_matrix_transposed = tfidf_matrix_transposed.T
#    ```
#    - *Transpose Option*: Allows analysis of terms instead of documents if set to `True`.
# 
# 2. *Applying PCA*:
#    The TF-IDF matrix is reduced to two dimensions using PCA:
#    ```python
#    pca = PCA(n_components=2)
#    pca_result = pca.fit_transform(tfidf_matrix_transposed)
#    ```
#    - *PCA*: Simplifies the high-dimensional data, making it easier to visualize document relationships.
# 
# 3. *Interactive Plot Creation*:
#    The script creates an interactive plot with Plotly:
#    ```python
#    fig = go.Figure()
#    for cluster_num in range(len(unique_clusters)):
#        cluster_docs_indices = [i for i, label in enumerate(domains_list) if label == unique_clusters[cluster_num]]
#        fig.add_trace(go.Scatter(...))
#    fig.show()
#    ```
#    - *Interactivity*: Users can hover over points to see document details, explore clusters, and analyze relationships.
# 
# **Importance and Relevance**
# 
# This function enables users to visualize and interact with complex text data, revealing clusters and patterns that are critical in LBD. It simplifies the process of identifying related documents, key topics, and potential new connections.
# 
# **Applications**
# 
# - *Academic Research*: Explore relationships between research papers.
# - *Market Analysis*: Identify trends in reports or reviews.
# - *Content Curation*: Organize and categorize large volumes of text.
# 
# **Usage**
# 
# Pass the TF-IDF matrix, document names, and domain information to the function to generate an interactive PCA plot. This visualization tool is essential for gaining insights into document relationships and clustering within large text corpora.

# In[7]:


def visualize_tfidf_pca_interactive(names, selected_names, domains_list, tfidf_matrix, transpose = False, color_schema = 0):
    tfidf_matrix_transposed = np.squeeze(np.asarray(tfidf_matrix))
    if transpose:
        tfidf_matrix_transposed = tfidf_matrix_transposed.T

    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(tfidf_matrix_transposed)
 
    # Generate colors for each point
    if color_schema == 0:
        colors = ['rgba(242, 69, 69, 0.2)', 'rgba(245, 233, 67, 0.2)', 'rgba(78, 222, 232, 0.2)', 'rgba(145, 232, 78, 0.2)',
                  'rgba(229, 2, 199, 0.2)', 'rgba(229, 145, 2, 0.2)', 'rgba(2, 229, 32, 0.2)', 'rgba(2, 85, 229, 0.2)']
        colors_centroid = ['rgba(242, 69, 69, 0.9)', 'rgba(245, 233, 67, 0.9)', 'rgba(78, 222, 232, 0.9)', 'rgba(145, 232, 78, 0.9)',
                    'rgba(229, 2, 199, 0.9)', 'rgba(229, 145, 2, 0.9)', 'rgba(2, 229, 32, 0.9)', 'rgba(2, 85, 229, 0.9)']
    elif color_schema == 1:
        colors = ['rgba(229, 2, 199, 0.2)', 'rgba(229, 145, 2, 0.2)', 'rgba(2, 229, 32, 0.2)', 'rgba(2, 85, 229, 0.2)',
                  'rgba(242, 69, 69, 0.2)', 'rgba(245, 233, 67, 0.2)', 'rgba(78, 222, 232, 0.2)', 'rgba(145, 232, 78, 0.2)']
        colors_centroid = ['rgba(229, 2, 199, 0.9)', 'rgba(229, 145, 2, 0.9)', 'rgba(2, 229, 32, 0.9)', 'rgba(2, 85, 229, 0.9)',
                    'rgba(242, 69, 69, 0.9)', 'rgba(245, 233, 67, 0.9)', 'rgba(78, 222, 232, 0.9)', 'rgba(145, 232, 78, 0.9)']
    elif color_schema == 2:
        colors = ['rgba(2, 229, 32, 0.2)', 'rgba(2, 85, 229, 0.2)', 'rgba(229, 2, 199, 0.2)', 'rgba(229, 145, 2, 0.2)', 
                  'rgba(242, 69, 69, 0.2)', 'rgba(245, 233, 67, 0.2)', 'rgba(78, 222, 232, 0.2)', 'rgba(145, 232, 78, 0.2)']
        colors_centroid = ['rgba(2, 229, 32, 0.9)', 'rgba(2, 85, 229, 0.9)', 'rgba(229, 2, 199, 0.9)', 'rgba(229, 145, 2, 0.9)', 
                    'rgba(242, 69, 69, 0.9)', 'rgba(245, 233, 67, 0.9)', 'rgba(78, 222, 232, 0.9)', 'rgba(145, 232, 78, 0.9)']
    elif color_schema == 11:
        colors = ['rgba(255, 0, 0, 0.2)', 'rgba(0, 0, 255, 0.2)']
        colors_centroid = ['rgba(255, 0, 0, 0.9)', 'rgba(0, 0, 255, 0.9)']
    elif color_schema == 12:
        colors = ['rgba(255, 153, 153, 0.2)', 'rgba(153, 153, 255, 0.2)']
        colors_centroid = ['rgba(255, 153, 153, 0.9)', 'rgba(153, 153, 255, 0.9)']
    elif color_schema == 13:
        colors = ['rgba(255, 0, 0, 0.2)', 'rgba(255, 128, 0, 0.2)', 'rgba(0, 0, 255, 0.2)', 'rgba(128, 0, 255, 0.2)']
        colors_centroid = ['rgba(255, 0, 0, 0.9)', 'rgba(255, 128, 0, 0.9)', 'rgba(0, 0, 255, 0.9)', 'rgba(128, 0, 255, 0.9)']
    else:
        colors = ['red', 'green', 'blue', 'yellow', 'black', 'grey', 'violet', 'brown', 'lime', 'cyan']

    # Determine unique clusters
    unique_clusters = list(set(domains_list))
    unique_clusters.sort()
    
    # Compute the centroid of the PCA result
    centroid = pca_result.mean(axis=0)
    
    # Create interactive plot
    fig = go.Figure()

    # PCA Scatter plot with random colors
    for cluster_num in range(len(unique_clusters)):
        cluster_docs_indices = [i for i, label in enumerate(domains_list) if label == unique_clusters[cluster_num]]

        # Compute centroid for the current cluster
        centroid_x = np.mean(pca_result[cluster_docs_indices, 0])
        centroid_y = np.mean(pca_result[cluster_docs_indices, 1])

        fig.add_trace(go.Scatter(x=pca_result[cluster_docs_indices, 0], y=pca_result[cluster_docs_indices, 1], 
                                 mode='markers+text',
                                 # marker=dict(size=8, color=colors[cluster_num]), # , color=colors[cluster_num]
                                 marker=dict(size=8, color=colors[cluster_num], symbol='circle', line=dict(color='rgba(0, 0, 0, 0.1)', width=1)), 
                                 name=unique_clusters[cluster_num],
                                 hovertext=[names[i] for i in cluster_docs_indices], # this text is shown on hover
                                 text='', # [names[i] for i in cluster_docs_indices], # this text is set to show always
                                 textposition='bottom center'))

        # Plot the centroid of the current cluster
        fig.add_trace(go.Scatter(x=[centroid_x], y=[centroid_y],
                                 mode='markers+text',
                                 marker=dict(size=16, color=colors_centroid[cluster_num], symbol='star', line=dict(color='rgba(0, 0, 0, 0.5)', width=2)),
                                 name='Centroid ' + unique_clusters[cluster_num],
                                 hovertext='Centroid of ' + unique_clusters[cluster_num],
                                 text=unique_clusters[cluster_num],
                                 textposition='bottom center'))

    if selected_names == []:
        special_cluster_docs_indices = []
    else:
        special_cluster_docs_indices = [i for i, label in enumerate(names) if label in selected_names]

    if selected_names != []:
        fig.add_trace(go.Scatter(x=pca_result[special_cluster_docs_indices, 0], y=pca_result[special_cluster_docs_indices, 1], 
                                    mode='markers+text',
                                    marker=dict(size=10, color='green', symbol='circle', line=dict(color='rgba(0, 0, 0, 1.0)', width=1)), 
                                    name='selected',
                                    hovertext=[names[i] for i in special_cluster_docs_indices], # this text is shown on hover
                                    text='', 
                                    textposition='bottom center'))

    # Plot the centroid of the whole set of documents
    fig.add_trace(go.Scatter(x=[centroid[0]], y=[centroid[1]],
                             mode='markers',
                             marker=dict(size=20, color='black', symbol='cross'),
                             name='The main centroid',
                             hovertext=['The main centroid']))

    fig.update_layout(title="PCA Visualization of TF-IDF Vectors",
                      hovermode='closest',
                      showlegend=True,
                      width=1100,  # Set the width of the figure
                      height=1100)  # Set the height of the figure
    
    fig.show()


# In[ ]:




