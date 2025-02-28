#!/usr/bin/env python
# coding: utf-8

# # 3. Feature Extraction Module
# 
# This module implements various feature extraction techniques like Bag of Words, TF-IDF, and word embeddings.

# In[1]:


import numpy as np
from gensim.models import Word2Vec, FastText
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict


# This function create_bag_of_words takes a list of preprocessed text documents as input and uses the scikit-learn library 
# to create a bag of words representation of the corpus. 
# The function first creates an instance of the sklearn.feature_extraction.text.CountVectorizer, 
# which is an implementation of the bag of words model.
# 
# The function then fits the vectorizer to the input corpus and transforms the corpus into a bag of words matrix using 
# the vectorizer.fit_transform() method. The matrix is stored in the Compressed Sparse Row (CSR) format (csr_matrix) 
# for efficient storage and computation.
# 
# Finally, the function returns a tuple containing the CountVectorizer instance and the bag of words matrix.
# 
# **Functionality**
# 
# This Python function, `create_bag_of_words`, generates a Bag of Words (BoW) representation of a text corpus. The function takes in a list of preprocessed text documents and converts them into a matrix where each row represents a document, and each column represents a word or a sequence of words (n-grams) found in the corpus. The matrix entries indicate the frequency of these words within the documents.
# 
# The Bag of Words model is a foundational technique in Natural Language Processing (NLP) and Text Mining. It transforms text data into a structured numerical format that machine learning models can process. This is particularly crucial for tasks like text classification, sentiment analysis, and information retrieval. The ability to convert text into a quantifiable form allows for more sophisticated analysis and discovery, such as identifying patterns, trends, or novel insights in a large corpus of literature—commonly referred to as Literature-Based Discovery (LBD).
# 
# In the context of LBD, the BoW model can be used to extract significant terms from scientific papers, patents, or other textual data. By analyzing the frequency and distribution of words across a corpus, researchers can identify new connections between seemingly unrelated pieces of information. This can lead to the discovery of novel hypotheses or previously overlooked relationships between concepts.
# 
# **Use**
# 
# 1. *Input parameters*:
#     - `corpus`: A list of preprocessed text documents. Each document is expected to be a string where preprocessing (such as lowercasing, removing punctuation, etc.) has already been applied.
#     - `ngram_size`: Specifies the maximum size of n-grams to consider. An n-gram is a contiguous sequence of `n` items from a given text. For example, with `ngram_size=2`, the function will consider both single words (unigrams) and pairs of words (bigrams).
#     - `min_df`: Sets a threshold for ignoring infrequent words. Words that appear in fewer documents than this threshold will be ignored. This can be a count (e.g., `min_df=2` means words must appear in at least two documents) or a proportion (e.g., `min_df=0.01` means words must appear in at least 1% of the documents).
# 
# 2. *Return values*:
#     - A list of words (or n-grams) that are the features of the BoW model.
#     - A matrix where each row corresponds to a document and each column corresponds to a word (or n-gram). The values in the matrix represent the frequency of the word in the corresponding document.
# 
# 3. *Example Usage*:
#     ```python
#     corpus = [
#         "natural language processing with python",
#         "text mining and information retrieval",
#         "introduction to natural language processing",
#     ]
#     words, bow_matrix = create_bag_of_words(corpus, ngram_size=1, min_df=1)
#     print("Words:", words)
#     print("Bag of Words Matrix:\n", bow_matrix)
#     ```
# 
# **Applications**
# 
# - *Text classification*: Convert documents into BoW format for input into machine learning models to classify them into categories like spam detection, sentiment analysis, or topic classification.
#   
# - *Document similarity*: Compute the similarity between documents based on their BoW representations, useful in information retrieval and clustering tasks.
# 
# - *Feature extraction for NLP tasks*: Use BoW as a feature set for more complex NLP tasks like named entity recognition, part-of-speech tagging, or word sense disambiguation.
# 
# The `create_bag_of_words` function is a simple yet powerful tool to transform text data into a structured numerical format. This transformation is a critical first step in many text mining and NLP tasks, enabling more advanced analysis and discovery processes. Whether you’re conducting sentiment analysis, categorizing documents, or exploring new hypotheses through Literature-Based Discovery, understanding and using the BoW model is essential.

# In[2]:


def create_bag_of_words(corpus: List[str], ngram_size=1, min_df=1, max_features=None) -> Tuple[List, np.ndarray]:
    """
    Create a bag of words representation of a text corpus.
    :param corpus: List[str], a list of preprocessed text documents
    :ngram_size: max ngram size, default = 1 meaning single words
    :min_df: ignore words that have a document frequency strictly lower than the given threshold, default = 1
             if the value is between 0 and 0.999, it ignores the words with relative document frequency strictly lower than the threshold
     :return: Tuple[List, Matrix], the List of words and the bow matrix
    """
    # Create an instance of the CountVectorizer from scikit-learn
    vectorizer = CountVectorizer(ngram_range=(1, ngram_size), min_df=min_df, max_features=max_features, dtype=np.int32)
    # vectorizer = CountVectorizer(ngram_range=(1, ngram_size), min_df=min_df, stop_words='english')

    # Fit the vectorizer to the corpus and transform the corpus into a bag of words matrix
    bag_of_words_matrix = vectorizer.fit_transform(corpus)

    # Return the vectorizer and the bag of words matrix as a tuple
    return vectorizer.get_feature_names_out().tolist(), bag_of_words_matrix.toarray()



# Helper functions for summarizing bag_of_words and tfidf matices.

# In[3]:


def word_is_nterm(word: str):
    return ' ' in word

def sum_count_documents_containing_each_word(word_list: List, bow_matrix: np.ndarray) -> Dict[str, int]:
    # number of documents containing each word
    word_counts = (bow_matrix > 0).sum(axis=0)
    words_dict = {}
    for word, count in zip(word_list, word_counts):
        words_dict[word] = count
    return words_dict

def sum_count_each_word_in_all_documents(word_list: List, any_matrix: np.ndarray) -> Dict[str, int]:
    # number of occurences of each word in all documents; sum of matrix columns
    word_counts = (any_matrix).sum(axis=0)
    if type(word_counts) == np.matrix:
        word_counts = word_counts.tolist()[0]
    words_dict = {}
    for word, count in zip(word_list, word_counts):
        words_dict[word] = count
    return words_dict

def max_tfidf_each_word_in_all_documents(word_list: List, tfidf_matrix: np.ndarray) -> Dict[str, float]:
    # max tfidf of each word in all documents; max element found in each matrix column
    word_counts = (tfidf_matrix).max(axis=0)
    if type(word_counts) == np.matrix:
        word_counts = word_counts.tolist()[0]
    words_dict = {}
    for word, count in zip(word_list, word_counts):
        words_dict[word] = count
    return words_dict

def sum_count_all_words_in_each_document(ids_list: List, any_matrix: np.ndarray) -> Dict[str, int]:
    # number of words in each document; sum of matrix rows
    word_counts = (any_matrix).sum(axis=1)
    if type(word_counts) == np.matrix:
        word_counts = word_counts.transpose().tolist()[0]
    words_dict = {}
    for id, count in zip(ids_list, word_counts):
        words_dict[id] = count
    return words_dict

def max_tfidf_all_words_in_each_document(ids_list: List, tfidf_matrix: np.ndarray) -> Dict[str, float]:
    # max tfidf of words in each document; max element found in each matrix row
    word_counts = (tfidf_matrix).max(axis=1)
    if type(word_counts) == np.matrix:
        word_counts = word_counts.transpose().tolist()[0]
    words_dict = {}
    for id, count in zip(ids_list, word_counts):
        words_dict[id] = count
    return words_dict



# Helper function for extracting sub-matrix from a given matrix (bag_of_words and tfidf).
# 

# In[4]:


def filter_matrix(ids_list: List, word_list: List, any_matrix: np.ndarray, filter_rows: List, filter_columns: List) -> Tuple[List, List, np.ndarray]:
    # filter the input matrix (according to filter_rows and filter_columns) into output matrix and preserve the order of filters in new output matrix 
    """
    Construct a sub-matrix from the given matrix based on filter_rows and filter_columns.
    
    Parameters:
    - matrix (numpy.ndarray): The original matrix.
    - filter_rows (list): List of row indices to include in the sub-matrix.
    - filter_columns (list): List of column indices to include in the sub-matrix.
    
    Returns:
    - numpy.ndarray: The sub-matrix.
    """
    new_ids_list = []
    for ind in filter_rows:
        new_ids_list.append(ids_list[ind])
    new_word_list = []
    for ind in filter_columns:
        new_word_list.append(word_list[ind])
    return new_ids_list, new_word_list, any_matrix[np.ix_(filter_rows, filter_columns)]

def filter_matrix_columns(word_list: List, any_matrix: np.ndarray, filter_rows: List, filter_columns: List) -> Tuple[List, np.ndarray]:
    # filter columns of the input matrix (according to filter_rows and filter_columns) into output matrix and preserve the order of filters in new output matrix 
    """
    Construct a sub-matrix from the given matrix based on filter_columns.
    
    Parameters:
    - matrix (numpy.ndarray): The original matrix.
    - filter_rows (list): List of row indices to include in the sub-matrix.
    - filter_columns (list): List of column indices to include in the sub-matrix.
    
    Returns:
    - numpy.ndarray: The sub-matrix.
    """
    new_word_list = []
    for ind in filter_columns:
        new_word_list.append(word_list[ind])
    return new_word_list, any_matrix[np.ix_(filter_rows, filter_columns)]


# This function create_tfidf takes a list of preprocessed text documents as input and uses the scikit-learn library 
# to create a Term Frequency-Inverse Document Frequency (TF-IDF) representation of the corpus. 
# The function first creates an instance of the sklearn.feature_extraction.text.TfidfVectorizer, 
# which is an implementation of the TF-IDF model.
# 
# The function then fits the vectorizer to the input corpus and transforms the corpus into a TF-IDF matrix using 
# the vectorizer.fit_transform() method. The matrix is stored in the Compressed Sparse Row (CSR) format (csr_matrix) 
# for efficient storage and computation.
# 
# Finally, the function returns a tuple containing the list of features *TfidfVectorizer.get_feature_names_out().tolist()* that correspond to the columns of the TF-IDF matrix and the TF-IDF matrix *tfidf_matrix*.
# 
# 

# In[5]:


def create_tfidf(corpus: List[str], ngram_size=1, min_df=1, max_features=None) -> Tuple[List, np.ndarray]:
    """
    Create a TF-IDF representation of a text corpus.
    :param corpus: List[str], a list of preprocessed text documents
    :ngram_size: max ngram size, default = 1 meaning single words
    :min_df: ignore words that have a document frequency strictly lower than the given threshold, default = 1
             if the value is between 0 and 0.999, it ignores the words with relative document frequency strictly lower than the threshold
    :return: TTuple[List, Matrix], the List of words and the tf-idf matrix
    """
    # Create an instance of the TfidfVectorizer from scikit-learn
    vectorizer = TfidfVectorizer(ngram_range=(1, ngram_size), min_df=min_df, max_features=max_features, dtype=np.float32)
    # vectorizer = TfidfVectorizer(ngram_range=(1, ngram_size), min_df=min_df, stop_words='english')

    # Fit the vectorizer to the corpus and transform the corpus into a TF-IDF matrix
    # tfidf_matrix = np.asarray(vectorizer.fit_transform(corpus))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Return the vectorizer and the TF-IDF matrix as a tuple
    return vectorizer.get_feature_names_out().tolist(), tfidf_matrix.todense()



# Cluster the input matrix (according to cluster_rows) into output matrix, where the number of rows is the number of clusters.

# In[6]:


def cluster_matrix(ids_list: List, word_list: List, any_matrix: np.ndarray, cluster_rows: List) -> np.ndarray:
    """
    Construct an agregate sub-matrix from the given matrix based on cluster_rows.
    cluster_rows contains integer numbers representing index of a cluster of the corresponding document.
    The clusters are represented by numbers 0, 1, 2, ... 
    
    Parameters:
    - matrix (numpy.ndarray): The original matrix.
    - filter_rows (list): List of row indices to include in the sub-matrix.
    - filter_columns (list): List of column indices to include in the sub-matrix.
    
    Returns:
    - numpy.ndarray: The sub-matrix with the row dimensionality of number of different clusters.
    """
    new_ids_list = []
    for ind in cluster_rows:
        new_ids_list.append(ids_list[ind])
    return any_matrix[np.ix_(cluster_rows, cluster_rows)]


# This function create_word_embeddings takes a list of tokenized text documents and an optional embedding method 
# (either 'word2vec' or 'fasttext', defaulting to 'word2vec') as input, and uses the gensim library to create 
# word embeddings for the documents. The function first trains a word embedding model using the specified method, 
# creating either a Word2Vec or FastText model with a vector size of 100, a window size of 5, a minimum word count of 1, 
# and 4 worker threads for parallelization.
# 
# The function then initializes an empty matrix with the shape (len(tokens_list), model.vector_size) to store the document embeddings. 
# For each document in the input tokens_list, the function calculates the document embedding by averaging the word embeddings 
# of each token in the document. This is done by first retrieving the word embeddings for each token using the trained 
# model's model.wv[token] attribute, then calculating the mean of these embeddings along axis 0.
# 
# Finally, the function returns a tuple containing the word embedding model (either a Word2Vec or FastText instance) 
# and the document embeddings matrix as a NumPy array.
# 
# 

# In[7]:


def create_word_embeddings(tokens_list: List[List[str]], embedding_method: str = 'word2vec') -> Tuple[BaseEstimator, np.ndarray]:
    """
    Create word embeddings for a list of tokenized documents using the specified embedding method.
    :param tokens_list: List[List[str]], a list of tokenized text documents
    :param embedding_method: str, the embedding method to use, either 'word2vec' or 'fasttext' (default: 'word2vec')
    :return: Tuple[BaseEstimator, np.ndarray], the word embedding model and the document embeddings matrix
    """
    # Train the word embedding model based on the specified method
    if embedding_method == 'word2vec':
        model = Word2Vec(tokens_list, vector_size=100, window=5, min_count=1, workers=4)
    elif embedding_method == 'fasttext':
        model = FastText(tokens_list, vector_size=100, window=5, min_count=1, workers=4)
    else:
        raise ValueError("Invalid embedding method. Please use either 'word2vec' or 'fasttext'.")

    # Initialize an empty matrix to store the document embeddings
    document_embeddings = np.zeros((len(tokens_list), model.vector_size))

    # Calculate the document embeddings by averaging the word embeddings of each token
    for i, tokens in enumerate(tokens_list):
        token_embeddings = np.array([model.wv[token] for token in tokens])
        document_embeddings[i] = token_embeddings.mean(axis=0)

    return model, document_embeddings


# In[ ]:




