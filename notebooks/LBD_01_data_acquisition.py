#!/usr/bin/env python
# coding: utf-8

# # 1. Module for data acquisition
# 
# This module is responsible for capturing and loading text data from various sources such as text files, CSV files or APIs.

# Import python libraries:

# In[ ]:


# <a href="https://colab.research.google.com/" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# &nbsp; TODO: chech the possibility for porting and activating the link</a>


# In[ ]:


import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

#import nltk
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.collocations import *

import requests
import time
import xml.etree.ElementTree as ET

from typing import List, Dict, Any
import logging


# This module implements five functions for capturing text data from various sources.
# 
# - load_data_from_file(file_path: str) -> List[str]:
# - load_data_from_csv(file_path: str, column_name: str) -> List[str]:
# - load_data_from_web(url: str) -> List[str]:
# - load_data_from_pubmed(api_endpoint: str, params: Dict[str, Any]) -> List[str]:
# 
# An additional helper function:
# 
# - convert_file_to_ascii_encoding(input_filename: str, output_filename: str) -> None:
# 
# can be used to convert text data from various different encodings to the Ascii encoding, which is normally supported by text processing libraries such as `nltk`.

# The `load_data_from_file` function is designed to load text data from a file and return it as a list of strings, where each string represents a line from the file. This function is essential for working with textual data stored in files, which is a common scenario in data processing, machine learning, and natural language processing (NLP) tasks.
# 
# **Functionality**
# 
# 1. *File existence check*: The function first checks if the specified file exists using the `os.path.exists()` method. If the file does not exist, it raises a `FileNotFoundError` to prevent further errors down the line.
# 
# 2. *Reading the file*: If the file exists, it is opened in read mode. The content of the file is read line by line using `file.readlines()`, which returns a list where each element corresponds to a line in the file.
# 
# 3. *Logging*: The function logs the number of lines loaded from the file using the `logging.info()` method. This is useful for tracking and debugging, especially when dealing with large files.
# 
# 4. *Return data*: Finally, the list of strings (each string is a line from the file) is returned for further processing.
# 
# **Use**
# 
# To use this function, simply pass the path of the text file you want to load:
# 
# ```python
# file_path = 'path/to/file.txt'
# lines = load_data_from_file(file_path)
# ```
# 
# This will return a list of strings, each representing a line of text from the file. You can then proceed with your data processing, whether it involves parsing, analysis, or feeding it into a machine learning model.
# 
# **Considerations**
# 
# - *File encoding*: The function currently opens the file with the default system encoding. If you're working with files in different encodings (like UTF-8), you may need to adjust the `open` function to handle these encodings explicitly.
# - *Error handling*: The function raises an error if the file does not exist.

# In[ ]:


def load_data_from_file(file_path: str) -> List[str]:
    """
    Load text data from a file.
    :param file_path: str, the path to the text file
    :return: List[str], a list of strings containing the text data, each string is one line 
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        
    with open(file_path, 'r', encoding='ascii', errors='ignore') as file:
        data = file.readlines()

    logging.info(f'Loaded {len(data)} lines from "{file_path}".')

    return data


# The `convert_file_to_ascii_encoding` function is designed to read the contents of a text file and convert it to ASCII encoding. The converted content is then saved to a new file. This function is particularly useful when dealing with text that may contain non-ASCII characters, which could cause compatibility issues in certain applications or systems.
# 
# In many data processing tasks, especially when dealing with legacy systems or specific text-based formats, ensuring that text data is in ASCII encoding is crucial. ASCII encoding is a character encoding standard that uses 7 bits to represent characters, which makes it highly compatible with older systems and simpler text processing pipelines.
# 
# **Functionality**
# 
# 1. *Reading the input file*: The function first opens the specified input file in read mode and reads its entire content into a string variable.
# 
# 2. *Converting to ASCII*: The content is then encoded into ASCII using the `.encode('ascii', errors='replace')` method. This step replaces any non-ASCII characters with a placeholder (usually `?`), ensuring that the resulting string is pure ASCII.
# 
# 3. *Saving the output*: Finally, the ASCII-encoded content is written to a new file, specified by the `output_filename` parameter, ensuring that the output is in ASCII format.
# 
# **Applications**
# 
# - *Data standardization*: Convert various text data sources to a uniform ASCII encoding, making it easier to process and analyze them together.
# - *Legacy system integration*: Prepare text files for integration in the systems that only support ASCII encoding.
# - *Text processing*: Simplify the handling of text data by converting non-ASCII characters, which might otherwise cause errors or require complex handling.
# 
# **Use**
# 
# To use this function, provide the path to the input file (the file you want to convert) and the path to the output file (where you want to save the converted text):
# 
# ```python
# input_file = 'path/to/input_file.txt'
# output_file = 'path/to/output_file.txt'
# convert_file_to_ascii_encoding(input_file, output_file)
# ```
# 
# This will create a new file at `output_file` that contains the ASCII-encoded content of the `input_file`.
# 
# **Considerations**
# 
# - *Error handling*: The strategy `errors='replace'` replaces non-ASCII characters with a `?`. This is a safe option, but you might lose some data (e.g. special characters or diacritics). If preserving these characters is important, you should consider alternative error handling strategies such as `ignore` or `xmlcharrefreplace`.
#   
# - *Use cases for ASCII encoding*: While ASCII encoding is widely supported, it is limited in terms of characters that can be displayed. For text data containing international characters, other encodings such as UTF-8 are more suitable unless you have certain restrictions that require ASCII.

# In[ ]:


def convert_file_to_ascii_encoding(input_filename: str, output_filename: str) -> None:
    """
    Read the contents of a file and save it with ASCII encoding.
    
    Parameters:
    - input_filename (str): The name of the file to be read.
    - output_filename (str): The name of the file where the ASCII-encoded content should be saved.
    """
    with open(input_filename, 'r', encoding='utf-8', errors='ignore') as file:
        contents = file.read()

    # Convert to ASCII and handle non-ASCII characters using 'replace' error strategy
    ascii_contents = contents.encode('ascii', errors='replace').decode('ascii')

    with open(output_filename, 'w', encoding='ascii') as file:
        file.write(ascii_contents)


# The `load_data_from_csv` function is designed to load text data from a specific column in a CSV (Comma-Separated Values) file. This function reads the CSV file, extracts the data from the specified column, and returns it as a list of strings. This is particularly useful for working with structured data where text is organized under specific columns.
# 
# CSV files are one of the most common formats for storing structured data, and they are widely used across various domains such as data analysis, machine learning, and natural language processing. 
# 
# **Functionality**
# 
# 1. *File existence check*: The function first checks whether the specified CSV file exists using `os.path.exists()`. If the file is not found, a `FileNotFoundError` is raised to alert the user.
# 
# 2. *Reading the CSV file*: The function then reads the CSV file using the `pandas.read_csv()` function, which loads the file into a DataFrame. The file is read with UTF-8 encoding and uses `;` as the separator. This separator can be adjusted depending on the CSV file's format.
# 
# 3. *Column existence check*: After loading the data, the function checks whether the specified column exists in the DataFrame. If the column is not found, a `ValueError` is raised, informing the user that the column name is incorrect or doesn't exist in the file.
# 
# 4. *Extracting data*: If the column is found, the function extracts the data from that column and converts it to a list of strings using the `.tolist()` method. This list is then returned for further processing or analysis.
# 
# **Use**
# 
# To use this function, specify the path to the CSV file and the name of the column from which you want to extract text data:
# 
# ```python
# file_path = 'path/to/your/data.csv'
# column_name = 'TextColumn'
# data = load_data_from_csv(file_path, column_name)
# ```
# 
# This will return a list of strings, where each string represents an entry from the specified column in the CSV file.
# 
# **Considerations**
# 
# - *CSV format*: Ensure that the separator (`sep`) used in `pd.read_csv()` matches the one used in your CSV file. The default here is `;`, which is common in some regions and formats, but many CSV files use `,` as the separator.
# 
# - *Error handling*: The function includes checks for both file existence and column existence, making it robust against common user errors. However, ensure that the CSV file is well-formed and that the column names are correctly specified.
# 
# - *Data types*: This function is specifically designed for loading text data. If the column contains other data types (e.g., numeric or mixed types), further processing might be required.

# In[ ]:


def load_data_from_csv(file_path: str, column_name: str) -> List[str]:
    """
    Load text data from a specific column in a CSV file.
    :param file_path: str, the path to the CSV file
    :param column_name: str, the name of the column containing the text data
    :return: List[str], a list of strings containing the text data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    df = pd.read_csv(file_path, encoding='utf-8', sep=';')
    
    if column_name not in df.columns:
        raise ValueError(f"The column '{column_name}' does not exist in the CSV file.")
        
    data = df[column_name].tolist()
    
    return data


# The function `load_data_from_web` was developed to scrape text content from a specific web page. It fetches the text within the `` tags of the HTML document and returns the content as a list of strings. This function is particularly useful for text mining and Literature-Based Discovery (LBD) to collect raw text data from online sources.
# 
# **Functionality**
# 
# 1. *Loading data from a URL*:
#    The function begins by making an HTTP GET request to the provided URL:
#    ```python
#    response = requests.get(url)
#    response.raise_for_status()
#    ```
#    - *HTTP request*: The `requests.get` method is used to fetch the content of the web page. If the request fails (e.g., due to network issues or an invalid URL), an exception is raised to handle the error gracefully.
# 
# 2. *Parsing the web page*:
#    The function then parses the HTML content of the page:
#    ```python
#    soup = BeautifulSoup(response.content, 'html.parser')
#    paragraphs = soup.find_all('p')
#    ```
#    - *BeautifulSoup*: This library is used to parse the HTML and locate all `<p>` (paragraph) tags, which typically contain the main textual content of the page.
# 
# 3. *Extracting and returning text*:
#    The text within each paragraph tag is extracted and returned as a list:
#    ```python
#    data = [paragraph.get_text() for paragraph in paragraphs]
#    return data
#    ```
#    - *Text extraction*: The `get_text()` method is used to extract the text from each paragraph, which is then stored in a list.
# 
# **Applications**
# 
# - *Content aggregation*: Collecting articles, blog posts or research papers for analysis.
# - *Sentiment analysis*: Collecting user reviews or posts on social media for sentiment analysis.
# - *Market research*: Scraping competitors' websites to analyze trends or extract relevant information.
# 
# **Use**
# 
# To use this function, simply enter the URL of the web page from which you want to retrieve data. The function will return a list of strings, each representing a section of text from the page. This data can then be further processed for your specific analysis requirements.

# In[ ]:


def load_data_from_web(url: str) -> List[str]:
    """
    Scrape text data from a web page.
    :param url: str, the URL of the web page
    :return: List[str], a list of strings containing the text data
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to load data from URL '{url}': {e}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    
    data = [paragraph.get_text() for paragraph in paragraphs]
    
    return data


# **Prepare parameters and class for retrieving articles from PubMed** 
# 
# Define the URL and the parameters to be used for retrieving articles from PubMed. Define the class `PubMedArticleRetriever` for searching (function `esearch`) and retrieving (function `efetch`) articles from PubMed.

# In[ ]:


# Base URL
URL_EUTILS = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
# Paths of associated e-utilities
URL_ESEARCH = 'esearch.fcgi?'
URL_EFETCH = 'efetch.fcgi?'

class PubMedArticleRetriever():
    # Args receives unlimited no. of arguments as an array
    def __init__(self, **kwargs):
        # Access args index like array does
        self.db = 'pubmed'
        
    def esearch(self, search_term, **kwargs):
        """Search Pubmed for paper IDs given a search term. ESearch Some useful
        parameters to pass are db='pmc' to search PMC instead... Note also
        the retstart argument along with retmax to page across batches of IDs.
        Parameters
        ----------
        search_term : str
            A term for which the PubMed search should be performed.
        kwargs : kwargs
            Additional keyword arguments to pass to the PubMed search as
            parameters.
        """
        params = {'db': 'pubmed',
                  'term': search_term,
                  'retmode': 'xml',
                  'rettype': 'uilist',
                  'retmax': 10000,
                  'usehistory': 'y'}
        params.update(kwargs)
        params = '&'.join('{}={}'.format(k, v) for k, v in params.items())
        url_call = URL_EUTILS + URL_ESEARCH + params
        resp = requests.get(url_call)
        root = ET.fromstring(resp.content)
        return root
    
    def efetch(self, pmid_lst, **kwargs):
        chunk_size = 100
        chunk_lst = [pmid_lst[x : x + chunk_size] for x in range(0, len(pmid_lst), chunk_size)]
        res_lst = []
        for i, chunk in enumerate(chunk_lst, start=1):
            logging.info(f'Fetching {chunk_size} articles of chunk: {i}/{len(chunk_lst)}')
            pmid_str = ','.join(pmid for pmid in chunk)
            params = {'db': 'pubmed',
                      'id': pmid_str,
                      'retmode': 'xml'}
            params.update(kwargs)
            params = '&'.join('{}={}'.format(k, v) for k, v in params.items())
            url_call = URL_EUTILS + URL_EFETCH + params
            resp = requests.get(url_call)
            root = ET.fromstring(resp.content)
            res_lst.append(root)
            time.sleep(3)
        return res_lst


# The function `convert_dict_to_list` was developed to convert the contents of a dictionary into a formatted list of character strings. The function combines the dictionary keys with specific values and a domain name to produce a structured output that is useful for further processing or storage.
# 
# **Applications**
# 
# - *Data export*: Convert structured data into a format suitable for exporting to text files or other systems.
# - *Data preprocessing*: Prepare data for text mining by converting it into a consistent format that can be easily tokenized and analyzed.
# - *Document management*: Organize and store metadata (e.g. titles, abstracts) from various documents in a human-readable format.
# 
# **Use**
# 
# To use this function, pass a dictionary in which each entry contains relevant document details, along with a string for the domain name. The function returns a list of formatted strings, each representing an entry in the dictionary. This output can then be used for further processing, storage or analysis.

# In[ ]:


def convert_dict_to_list(dictionary: Dict[str, Any], domain_name: str) -> List[str]:
    """
    Converts the contents of a dictionary to a list of strings.
    
    :param dictionary: The dictionary to be converted.
    :param domain_name: The name of domain to be included in the file.
    """
    lines = []
    for key, value in dictionary.items():
        lines.append(key + ': !' + domain_name + ' ' + value['title'] + ' ' + value['abstract'])
    return lines


# The function `load_data_from_pubmed` was developed to retrieve and process text data from PubMed, a widely used database for biomedical literature. The function retrieves articles based on a search term and date range, extracts relevant information (such as titles and abstracts) and then formats this data into a list of strings. This functionality is of great importance for LBD, as access to large, domain-specific data sets is crucial.
# 
# **Functionality**
# 
# 1. *Initializing the PubMed article retriever*:
#    The function begins by creating an instance of `PubMedArticleRetriever`, a tool for querying PubMed.
# 
# 2. *Performing a PubMed search*:
#    The function searches PubMed for articles matching the specified criteria:
#    - *Search term*: The `search_str` parameter specifies the query (e.g., keywords or phrases).
#    - *Date range*: The `min_date` and `max_date` parameters define the publication date range for the search.
# 
# 3. *Retrieving PMIDs*:
#    The function extracts PubMed IDs (PMIDs) for the articles found in the search:
#    - *PMIDs*: Unique identifiers for PubMed articles, used to fetch detailed information.
# 
# 4. *Fetching article details*:
#    The function retrieves the detailed information (title and abstract) for each article:
#    - *Title and abstract extraction*: The function extracts and stores the title and abstract for each article, handling cases where the abstract might be missing.
# 
# 5. *Converting to a List of strings*:
#    The extracted data is then converted into a list of formatted strings using the `convert_dict_to_list` function.
# 
# **Meaning**
# 
# Programmatic access to and processing of PubMed data enables researchers to automate the retrieval of large volumes of biomedical literature, saving time and ensuring a comprehensive data set. This is essential for conducting thorough literature searches or for training machine learning models on domain-specific data.
# 
# **Applications**
# 
# - *Biomedical research*: Automate the search for relevant studies for systematic reviews or meta-analyzes.
# - *Drug discovery*: Collect literature on specific compounds or diseases to identify potential new drug targets.
# - *Scientific writing*: Collect a large number of references on a specific topic to review or cite in scientific papers.
# 
# **Use**
# 
# To use this function, enter a search term, a date range and a domain name. The function will return a list of formatted strings containing the titles and abstracts of the articles retrieved from PubMed. This list can then be used for further analysis, e.g. text mining, summarization or other LBD tasks.
# 
# For example, to retrieve articles from PubMed about *migraine* and *magnesium* in the title or abstract, from the year 1988, with the domain name *mig_mag* (stored in the line as *!mig_mag*), the call of the function is as follows:
# ```python
# lines = load_data_from_pubmed('migraine[tiab] AND magnesium[tiab]', '1988/01/01', '1988/12/31', 'mig_mag')
# ```

# In[ ]:


def load_data_from_pubmed(search_str: str, min_date: str, max_date: str, domain_name: str) -> List[str]:
    """
    Retrieve text data from PubMed.
    :param search_str: str, a string for search parameter,
    :param min_date: str, the minimum date for search parameter, format: YYYY/MM/DD,
    :param domain_name: str, domain name to be used to identify the documents,
    :return: List[str], a list of strings containing the text data
    """
    
    # 1. nitializing the PubMed article retriever
    pubmedAR = PubMedArticleRetriever()
    
    # 2. Performing a PubMed search
    esearch_root = pubmedAR.esearch(search_term = search_str, datetype = 'pdat', mindate = min_date, maxdate = max_date)

    # 3. Retrieving PMIDs
    pmid_lst = [id.text for id in esearch_root.findall('IdList/Id')]
    logging.info(f'Number of PMIDs: {len(pmid_lst)}')
    logging.info(f'List of PMIDs: {pmid_lst}')

    # 4. Fetching article details
    efetch_lst = pubmedAR.efetch(pmid_lst)
 
    c_dct = {}
    for chunk in efetch_lst:
        for medline_citation in chunk.findall('.//MedlineCitation'):
            pmid = medline_citation.find('PMID')
            article = medline_citation.find('./Article')
            title = article.find('ArticleTitle')
            abstracte = article.find('Abstract/AbstractText')
            if abstracte is None:
                abstract = ''
            else:
                abstract = abstracte.text
            c_dct[pmid.text] = {'title': title.text, 'abstract': abstract} 

    # 5. Converting to a List of strings
    lines = convert_dict_to_list(c_dct, domain_name)

    return lines


# In[ ]:




