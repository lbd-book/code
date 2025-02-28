#!/usr/bin/env python
# coding: utf-8

# # 2. Data preprocessing module
# 
# This module performs the preprocessing tasks for text mining such as tokenization, stop word removal, stemming and lemmatization.
# 
# The principles and functions of text preprocessing are generally described in the accompanying book. They are also partly based on the following books, which are also recommended for further reading:
# 
# <hr>
# 
# [1] Kedia, A., Rasu, M. (2020): Hands-On Python Natural Language Processing: Explore tools and techniques to analyze and process text with a view to building real-world NLP applications, Pakt Publishing.
# 
# [2] Antić, Z. (2021): Python Natural Language Processing Cookbook: Over 50 recipes to understand, analyze, and generate text for implementing language processing tasks, Pakt Publishing.
# 

# In[1]:


from typing import List, Dict

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import re
import spacy

nlp2 = spacy.load("en_core_web_md")

import logging


# In[2]:


# Define string qualifiers that are used in the following functions

strDomainDefault = 'NA' # 'default'
strIdDefault = 'tmp'
strDomainKey = 'domain'
strDocumentKey = 'document'
strPreprocessedKey = 'preprocessed'
strPreprocessedDefaultValue = ['NA']


# A function for displaying longer strings truncated to a specified length with added ellipsis (...) if it's longer than the specified length.

# In[5]:


def truncate_with_ellipsis(string: str, length: int) -> str:
    """
    Truncate a string to a specified length and add ellipsis (...) if it's longer than the specified length.
    
    Parameters:
    - string (str): the string to be truncated
    - length (int): the maximum allowed length of the truncated string
    
    Returns:
    - str: the truncated string with ellipsis (...) if truncation occurred
    """
    if len(string) > length:
        return string[:length - 3] + '...'  # Adjust for the length of the ellipsis
    return string


# The fuction *construct_dict_from_list* is designed to process a list of documents *docs_list* and transform them into a structured dictionary format. Each document in the list is expected to potentially have an identifier *id:* and a domain *!domain* at the beginning, followed by the main content of the document. The procedure aims to extract these components and store them in a structured manner within a dictionary. The structure of the returned dictionary is the following:
# <code>
# {pubmed_id: {domain: 'str', document: 'str', preprocessed: 'str'}, ...
# }</code>

# In[6]:


def construct_dict_from_list(docs_list: List[str]) -> Dict[str, Dict]:
    """
    Transform a list of documents (strings) to dictionary, such that 
    the first word in the document (if terminated by :) becomes a key label in the dictionary
    and the second word in the document (if preceeded by !) becomes a domain label in the subdictionary
    end the rest of the document becomes a document string in the subdictionary
    :param docs_list: List[str], a list of documents
    :return: Dict[str, Dict], a dictionary where the first word is the key to subdictionary with domain name and document text; 
             in subdictionary there is also a placeholder for preprocessed text
    """
    processed_dict = {}
    tmp_id = -1
    for doc in docs_list:
        tmp_id += 1
        # Extract the original id: and !domain from the document doc, if they exist
        tokens = doc.split()
        tmp_domain = strDomainDefault

        doc_id = tokens[0]
        if doc_id[-1] == ':':
            pubmed_id = doc_id[:-1]
            tokens.pop(0)
        else:
            pubmed_id = strIdDefault + str(tmp_id)

        doc_domain = tokens[0]
        if doc_domain[0] == '!':
            pubmed_domain = doc_domain[1:]
            tokens.pop(0)
        else:
            pubmed_domain = tmp_domain

        # Store the domain name and the document in a new dictionary
        processed_dict[pubmed_id] = {
            strDomainKey: pubmed_domain,
            strDocumentKey: ' '.join(tokens),
            strPreprocessedKey: strPreprocessedDefaultValue
        }
    return processed_dict


# The function *construct_dict_from_lists* constructs a document dictionary from lists of ids, domains and documents (original and preprocessed). All the input lists are of the same length; the order of the items in the input lists are ordered correspondingly to the list of documents *docs_list*.

# In[7]:


# Construct a document dictionary from lists of ids, domains and documents (original and preprocessed)
def construct_dict_from_lists(ids_list: List[str], domains_list: List[str], docs_list: List[str], prep_docs_list: List[str]) -> Dict[str, Dict]:
    """
    Construct a dictionary from three lists: ids, domains and docs, where all the lists are of same length and
    the list items with the same index correspond to the same document
    :param ids_list: List[str], a list of ids (e.g. pubmed_ids)
    :param domains_list: List[str], a list of domains
    :param docs_list: List[str], a list of documents
    :return: Dict[str, Dict], a dictionary where the first word is the key to subdictionary with domain name and document text
    """
    processed_dict = {}
    for num in range(len(ids_list)):
        # Combine and store the list elements to the dictionary
        processed_dict[ids_list[num]] = {
            strDomainKey: domains_list[num],
            strDocumentKey: docs_list[num],
            strPreprocessedKey: prep_docs_list[num]
        }
    return processed_dict


# The foolowing functions are used to extract list (*ids_list*, *domain_names_list*, *documents_list* and *preprocessed_documents_list*) from the dictionary *docs_dict*. 

# In[11]:


def extract_ids_list(docs_dict: Dict[str, Dict]) -> List[str]:
    processed_list = []
    for pubmed_id, doc in docs_dict.items():
        # Extract the id (key) from the document dictionary
        processed_list.append(pubmed_id)
    return processed_list

def extract_domain_names_list(docs_dict: Dict[str, Dict]) -> List[str]:
    processed_list = []
    for pubmed_id, doc in docs_dict.items():
        # Extract the original domain name from the document dictionary
        processed_list.append(doc[strDomainKey])
    return processed_list

def extract_unique_domain_names_list(docs_dict: Dict[str, Dict]) -> List[str]:
    processed_list = []
    for pubmed_id, doc in docs_dict.items():
        # Extract the original domain name from the document dictionary and store in not already present
        if doc[strDomainKey] not in processed_list:
            processed_list.append(doc[strDomainKey])
    return processed_list

def extract_documents_list(docs_dict: Dict[str, Dict]) -> List[str]:
    processed_list = []
    for pubmed_id, doc in docs_dict.items():
        # Extract the original document string from the the document dictionary
        processed_list.append(doc[strDocumentKey])
    return processed_list

def extract_preprocessed_documents_list(docs_dict: Dict[str, Dict]) -> List[str]:
    processed_list = []
    for pubmed_id, doc in docs_dict.items():
        # Extract the original document string from the the document dictionary
        processed_list.append(doc[strPreprocessedKey])
    return processed_list


# Helper functions for saving lists and dictionaries to text files for further documentation and inspection.

# In[12]:


def save_list_to_file(my_list: List, filename: str):
    """
    Save the provided list to a text file.
    
    Parameters:
    - my_list (list): The list of elements to be saved.
    - filename (str): The name of the file where the list should be saved.
    """
    with open(filename, 'w') as file:
        for item in my_list:
            file.write("%s\n" % item)

def sort_dict_by_value(d: Dict, reverse = False) -> Dict:
    """
    Sort a dictionary by the value in ascending (default) or descending (reverse = True) order.
    The values in the dictionary should be elementary (int, float, str)
    
    Parameters:
    - d (dict): The dictionary of elements to be sorted.
    - reverse (boolean): The reverse order of sorting.
    """
    return dict(sorted(d.items(), key = lambda x: x[1], reverse = reverse))

def get_index_list_of_dict1_keys(dict1: Dict, list2: List):
    """
    Return a list of indeces for dict1 keys in the second dictionary dict2.
    
    Parameters:
    - dict1, dict2 (dict): The two dictionaries.
    """
    ind_list = []
    for key, value in dict1.items():
        ind_list.append(list2.index(key))
    return ind_list


# This function cleans text.

# In[17]:


def do_clean_text(corpus, keep_list, remove_list):
    """
    Purpose: function to keep only alphabets; short words (single character), digits and certain words (punctuations, tabs etc.) removed
    
    Input: a text corpus, 'corpus' to be cleaned along with a list of words, 'keep_list', which have to be retained
           even after the cleaning process, and words in 'remove_list', which have to be removed unconditionally 
    
    Output: the cleaned text corpus
    
    """
    cleaned_corpus = []
    for row in corpus:
        qs = []
        for word in row.split():
            word = word.lower()
            # the pattern in re.sub determines which characters are accepted as valid 
            pattern = '[^a-z0-9čšž]'
            p1 = re.sub(pattern, '', word)
            if p1 not in keep_list:
                if p1 not in remove_list:
                    if (not p1.isdigit()) and (len(p1) >= 2):
                        qs.append(p1)
            else:
                qs.append(p1)
        cleaned_corpus.append(' '.join(qs))
    return cleaned_corpus


# In[19]:


# This function is partially based on [1]
def do_remove_stopwords(corpus):
    wh_words = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']
    stop = set(stopwords.words('english'))
    for word in wh_words:
        stop.remove(word)
    corpus = [[x for x in x.split() if x not in stop] for x in corpus]
    return corpus


# In[20]:


# This function is partially based on [1]
def do_lemmatize(corpus):
    lem = WordNetLemmatizer()
    corpus = [[lem.lemmatize(x, pos = 'v') for x in x] for x in corpus]
    return corpus


# In[21]:


# This function is partially based on [1]
def do_stem(corpus, stem_type = None):
    if stem_type == 'snowball':
        stemmer = SnowballStemmer(language = 'english')
        corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    else :
        stemmer = PorterStemmer()
        corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    return corpus


# In[22]:


def do_keep_only_longer_or_equal_length_words(corpus, min_length):
    modified_corpus = []
    for sublist in corpus:
        modified_sublist = []
        for term in sublist:
            # if (len(term) >= min_length) and (not term[0].isdigit()):
            if (len(term) >= min_length):
                modified_sublist.append(term)
        modified_corpus.append(modified_sublist)
    return modified_corpus


# In[23]:


def do_remove_non_nouns(corpus, keep_list):
    modified_corpus = []
    for sublist in corpus:
        modified_sublist = []
        doc = nlp2(' '.join(sublist))
        pos_terms = [(token.text, token.pos_) for token in doc]
        for term in pos_terms:
            # if (term[1] == 'NOUN') or (term[1] == 'PROPN') or (term[1] == 'ADJ') or (term[1] == 'VERB'):
            if term[0] in keep_list:
                modified_sublist.append(term[0])
            elif (term[1] == 'NOUN') or (term[1] == 'PROPN'):
                modified_sublist.append(term[0])
        modified_corpus.append(modified_sublist)
    return modified_corpus


# In[24]:


def do_remove_non_mesh(corpus, mesh_word_list):
    modified_corpus = []
    for sublist in corpus:
        modified_sublist = []
        for term in sublist:
            if term in mesh_word_list:
                modified_sublist.append(term)
        modified_corpus.append(modified_sublist)
    return modified_corpus


# In[25]:


# This function is partially based on [1]
def preprocess(corpus, keep_list, remove_list, mesh_word_list, \
               cleaning = True, remove_stopwords = True, lemmatization = True, \
               min_word_length = 0, keep_only_nouns = False, keep_only_mesh = False, stemming = False, stem_type = None):
    """
    Purpose : Function to perform all pre-processing tasks (cleaning, stemming, lemmatization, stopwords removal etc.)
    
    Input : 
    'corpus' - Text corpus on which pre-processing tasks will be performed - in the form of a list of strings - documents
    'keep_list' - List of words to be retained during cleaning process
    'min_word_length' - minimal length of word to be kept after the preprocessing; default 0 means all the wors are kept
    'cleaning', 'remove_stopwords', 'lemmatization', 'stemming', 'keep_only_nouns' - Boolean variables indicating whether 
                                                                  a particular task should be performed or not
    'stem_type' - Choose between Porter stemmer or Snowball(Porter2) stemmer. Default is "None", which corresponds to Porter
                  Stemmer. 'snowball' corresponds to Snowball Stemmer
    
    Note : Either stemming or lemmatization should be used. There's no benefit of using both of them together
    
    Output : Returns the preprocessed text corpus - in the form of a list of documents (each document is a string of words)
    
    """

    if cleaning == True:
        logging.info('Text cleaning ...')
        corpus = do_clean_text(corpus, keep_list, remove_list)
    
    if remove_stopwords == True:
        logging.info('Removing stopwords ...')
        corpus = do_remove_stopwords(corpus)
    else:
        corpus = [[x for x in x.split()] for x in corpus]
    
    if lemmatization == True:
        logging.info('Lemmatization ...')
        corpus = do_lemmatize(corpus)

    if min_word_length > 0:
        logging.info('Keeping only longer words (>= ' + str(min_word_length) + ' characters)...')
        corpus = do_keep_only_longer_or_equal_length_words(corpus, min_word_length) # add keep_list

    if keep_only_nouns:
        logging.info('Keeping only nouns ...')
        corpus = do_remove_non_nouns(corpus, keep_list) # add keep_list

    if keep_only_mesh:
        logging.info('Keeping only selected MeSH terms ...')
        corpus = do_remove_non_mesh(corpus, mesh_word_list) # add keep_list

    if stemming == True:
        logging.info('Stemming ...')
        corpus = do_stem(corpus, stem_type)

    corpus = [' '.join(x) for x in corpus]  
    logging.info('Preprocessing finished.')      

    return corpus


# The *preprocess_docs_dict* function performs various pre-processing tasks on a dictionary of documents. The function first extracts the IDs, domain names, and actual documents from the docs_dict. It then preprocesses the text by potentially cleaning, stemming, lemmatizing, and removing stopwords based on the provided boolean flags. The processed corpus is then combined back with the IDs and domain names to form a new dictionary that is returned by the function.
# 
# Note: One should either use stemming or lemmatization, but not both, as there is no added benefit to using them together.

# In[27]:


def preprocess_docs_dict(docs_dict, keep_list, remove_list, mesh_word_list, \
               cleaning = True, remove_stopwords = True, lemmatization = True, \
               min_word_length = 0, keep_only_nouns = False, keep_only_mesh = False, stemming = False, stem_type = None):
    """
    Purpose : Perform the pre-processing tasks (cleaning, stemming, lemmatization, stopwords removal) on the docs_dict
    
    Input : 
    'docs_dict' - Dictionary of documents on which the preprocessing tasks are performed
    'keep_list' - List of words to be retained during cleaning process
    'cleaning', 'stemming', 'lemmatization', 'remove_stopwords' - Boolean variables indicating whether a particular task should 
                                                                  be performed or not
    'stem_type' - Choose between Porter stemmer or Snowball (Porter2) stemmer. Default is "None", which corresponds to Porter
                  Stemmer. 'snowball' corresponds to Snowball Stemmer
    
    Output : Returns the processed corpus in the form of a dictionary with the structure descibed earlier.

    """
    
    ids_list = extract_ids_list(docs_dict)
    domains_list = extract_domain_names_list(docs_dict)
    corpus = extract_documents_list(docs_dict)

    corpus_with_preprocessing = preprocess(corpus, keep_list = keep_list, remove_list = remove_list, mesh_word_list = mesh_word_list, \
                cleaning = cleaning, remove_stopwords = remove_stopwords, lemmatization = lemmatization, \
                min_word_length = min_word_length, keep_only_nouns = keep_only_nouns, keep_only_mesh = keep_only_mesh, \
                stemming = stemming, stem_type = stem_type)

    return construct_dict_from_lists(ids_list, domains_list, corpus, corpus_with_preprocessing)


# In[ ]:




