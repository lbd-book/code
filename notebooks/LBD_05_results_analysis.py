#!/usr/bin/env python
# coding: utf-8

# # 5. Results Analysis Module
# 
# This module helps analyze the results from the text mining algorithms and provide insights into the data.

# In[1]:


from typing import List, Tuple, Dict
from sklearn.base import BaseEstimator
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances


# This function analyze_topics takes a trained topic model (such as an LDA model), a list of feature names 
# (for example, the words from a CountVectorizer or TfidfVectorizer), and an integer specifying the number 
# of top words to display for each topic as input. The function analyzes the topics extracted from the topic model 
# by listing the top words for each topic.
# 
# The function initializes an empty list topics to store the results. The function then iterates through the components 
# of the topic model using a for loop with the enumerate() function, which provides both the index and the component (topic) 
# for each iteration. For each topic, the function gets the indices of the top words by sorting the topic's word weights 
# in ascending order and then slicing the result to obtain the n_top_words highest-weighted words 
# using the [:-n_top_words - 1:-1] slice notation.
# 
# The function then retrieves the feature names (words) for the top words using a list comprehension that maps 
# the top word indices to the corresponding feature names in the feature_names input list. 
# The function appends a tuple containing the topic index and the top words to the topics list.
# 
# Finally, the function returns the list of tuples containing the topic index and the top words for each topic.
# 
# 

# In[2]:


def analyze_topics(topic_model: BaseEstimator, feature_names: List[str], n_top_words: int) -> List[Tuple[int, List[str]]]:
    """
    Analyze the topics extracted from a topic model by listing the top words for each topic.
    :param topic_model: BaseEstimator, a trained topic model (e.g., LDA model)
    :param feature_names: List[str], a list of feature names (e.g., words from a CountVectorizer or TfidfVectorizer)
    :param n_top_words: int, the number of top words to display for each topic
    :return: List[Tuple[int, List[str]]], a list of tuples containing the topic index and the top words for each topic
    """
    # Initialize an empty list to store the results
    topics = []

    # Iterate through the components of the topic model
    for topic_idx, topic in enumerate(topic_model.components_):
        # Get the indices of the top words for the current topic
        top_word_indices = topic.argsort()[:-n_top_words - 1:-1]

        # Get the feature names (words) for the top words
        top_words = [feature_names[i] for i in top_word_indices]

        # Append the topic index and top words as a tuple to the topics list
        topics.append((topic_idx, top_words))

    return topics



# This function analyze_sentiment takes a list of sentiment scores for each document as input and analyzes the sentiment scores 
# by calculating the average sentiment, and the proportion of positive, negative, and neutral documents. 
# The function first calculates the average sentiment by summing up the sentiment scores and dividing the result 
# by the number of documents (i.e., the length of the sentiment_scores list).
# 
# The function then calculates the proportions of positive, negative, and neutral documents by counting 
# the number of documents with sentiment scores greater than 0.05, less than -0.05, and between -0.05 and 0.05 (inclusive), 
# respectively, and dividing the counts by the number of documents. These calculations use list comprehensions 
# to filter the sentiment_scores list based on the sentiment score thresholds.
# 
# Finally, the function returns the results as a dictionary containing the average sentiment, 
# and the proportion of positive, negative, and neutral documents.
# 
# 

# In[3]:


def analyze_sentiment(sentiment_scores: List[float]) -> Dict[str, float]:
    """
    Analyze the sentiment scores by calculating the average sentiment, and the proportion of positive, negative, and neutral documents.
    :param sentiment_scores: List[float], a list of sentiment scores for each document
    :return: Dict[str, float], a dictionary containing the average sentiment, and the proportion of positive, negative, and neutral documents
    """
    # Calculate the average sentiment
    average_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    # Calculate the proportions of positive, negative, and neutral documents
    positive_proportion = len([score for score in sentiment_scores if score > 0.05]) / len(sentiment_scores)
    negative_proportion = len([score for score in sentiment_scores if score < -0.05]) / len(sentiment_scores)
    neutral_proportion = len([score for score in sentiment_scores if -0.05 <= score <= 0.05]) / len(sentiment_scores)

    # Return the results as a dictionary
    return {
        'average_sentiment': average_sentiment,
        'positive_proportion': positive_proportion,
        'negative_proportion': negative_proportion,
        'neutral_proportion': neutral_proportion,
    }



# This function analyze_clusters takes a trained clustering model (such as a KMeans model) and a document-term matrix in 
# the Compressed Sparse Row (CSR) format as input. The function analyzes the clusters created by the clustering model 
# by listing the indices of the documents in each cluster.
# 
# The function first obtains the cluster labels for each document using the clustering model's predict() method, 
# which assigns each document in the input matrix to a cluster. The function initializes an empty dictionary clusters to store the results.
# 
# The function then iterates through the unique cluster labels using a for loop with the set() function, 
# which removes duplicate labels. For each cluster label, the function gets the indices of the documents 
# in the current cluster by using a list comprehension that filters the cluster_labels list based on the current label. 
# The function adds the document indices to the clusters dictionary with the cluster label as the key.
# 
# Finally, the function returns the dictionary containing the cluster index as the key and the list of document indices 
# in each cluster as the value.
# 
# 

# In[4]:


def analyze_clusters(clustering_model: BaseEstimator, matrix: csr_matrix) -> Dict[int, List[int]]:
    """
    Analyze the clusters created by a clustering model by listing the indices of the documents in each cluster.
    :param clustering_model: BaseEstimator, a trained clustering model (e.g., KMeans)
    :param matrix: csr_matrix, the document-term matrix used for clustering
    :return: Dict[int, List[int]], a dictionary containing the cluster index as the key and the list of document indices in each cluster as the value
    """
    # Obtain the cluster labels for each document using the clustering model's predict method
    cluster_labels = clustering_model.predict(matrix)

    # Initialize an empty dictionary to store the results
    clusters = {}

    # Iterate through the unique cluster labels
    for label in set(cluster_labels):
        # Get the indices of the documents in the current cluster
        document_indices = [idx for idx, cluster_label in enumerate(cluster_labels) if cluster_label == label]

        # Add the document indices to the clusters dictionary with the cluster label as the key
        clusters[label] = document_indices

    return clusters



# This function compute_similarity takes a document-term matrix in the Compressed Sparse Row (CSR) format and a string 
# specifying the similarity metric to use as input. The function computes the similarity between documents 
# in the document-term matrix using the specified similarity metric.
# 
# The function first checks the input similarity_metric and computes the similarity matrix accordingly. 
# If the similarity metric is 'cosine', the function computes the similarity matrix using the cosine_similarity function 
# from scikit-learn's sklearn.metrics.pairwise module. If the similarity metric is 'euclidean', 
# the function computes the similarity matrix using the euclidean_distances function and then 
# normalizes the distances into the range (0, 1) by applying the following transformation: 1 / (1 + distances). 
# If the similarity metric is 'manhattan', the function computes the similarity matrix using 
# the manhattan_distances function and normalizes the distances in the same way as for the Euclidean distance. 
# If the input similarity metric is not one of these allowed values, the function raises a ValueError.
# 
# Finally, the function returns the similarity matrix as a NumPy array with shape (n_documents, n_documents), 
# where n_documents is the number of documents in the input document-term matrix.
# 
# 

# In[5]:


def compute_similarity(matrix: csr_matrix, similarity_metric: str = 'cosine') -> np.ndarray:
    """
    Compute the similarity between documents in a document-term matrix using a specified similarity metric.
    :param matrix: csr_matrix, the document-term matrix
    :param similarity_metric: str, the similarity metric to use (e.g. 'cosine', 'euclidean', or 'manhattan')
    :return: np.ndarray, the similarity matrix with shape (n_documents, n_documents)
    """
    # Check the input similarity_metric and compute the similarity matrix accordingly
    if similarity_metric == 'cosine':
        similarity_matrix = cosine_similarity(matrix)
    elif similarity_metric == 'euclidean':
        similarity_matrix = 1 / (1 + euclidean_distances(matrix))
    elif similarity_metric == 'manhattan':
        similarity_matrix = 1 / (1 + manhattan_distances(matrix))
    else:
        raise ValueError("Invalid similarity metric. Allowed values are 'cosine', 'euclidean', or 'manhattan'.")

    return similarity_matrix



# This function find_most_similar_documents takes a similarity matrix as a NumPy array with shape (n_documents, n_documents), 
# the index of the document for which to find the most similar documents, and an integer specifying the number of 
# most similar documents to return as input. The function finds the most similar documents to the specified document 
# using the similarity matrix.
# 
# The function first gets the similarity scores for the specified document by indexing the similarity_matrix with the document_index. 
# The function then gets the indices of the top_n most similar documents by sorting the similarity scores in ascending order, 
# slicing the result to obtain the top_n + 1 highest similarity scores (including the specified document itself) 
# using the [-top_n - 1:-1] slice notation, and then reversing the order of the indices using the [::-1] slice notation.
# 
# The function gets the similarity scores of the most similar documents by indexing the similarity_scores 
# with the most_similar_indices. The function then combines the indices and scores into a list of tuples using the zip() function 
# and the list() constructor.
# 
# Finally, the function returns the list of tuples containing the indices and similarity scores of the most similar documents.
# 
# 

# In[6]:


def find_most_similar_documents(similarity_matrix: np.ndarray, document_index: int, top_n: int = 5) -> List[Tuple[int, float]]:
    """
    Find the most similar documents to a specified document using a similarity matrix.
    :param similarity_matrix: np.ndarray, the similarity matrix with shape (n_documents, n_documents)
    :param document_index: int, the index of the document for which to find the most similar documents
    :param top_n: int, the number of most similar documents to return (default: 5)
    :return: List[Tuple[int, float]], a list of tuples containing the indices and similarity scores of the most similar documents
    """
    # Get the similarity scores for the specified document
    similarity_scores = similarity_matrix[document_index]

    # Get the indices of the top_n most similar documents (excluding the specified document itself)
    most_similar_indices = np.argsort(similarity_scores)[-top_n - 1:-1][::-1]

    # Get the similarity scores of the most similar documents
    most_similar_scores = similarity_scores[most_similar_indices]

    # Combine the indices and scores into a list of tuples
    most_similar_documents = list(zip(most_similar_indices, most_similar_scores))

    return most_similar_documents


# The Results Analysis Module provides functions to extract insights from the output of text mining algorithms. The module allows users to analyze topics from topic modeling, calculate summary statistics for sentiment scores, analyze document clusters, compute document similarity, and find the most similar documents to a given document. These functions help users better understand the patterns and trends in their text data, ultimately facilitating data-driven decision-making.
# 

# In[ ]:




