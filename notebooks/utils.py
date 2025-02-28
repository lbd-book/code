import gzip
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import groupby, filterfalse
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


# https://www.nlm.nih.gov/research/umls/knowledge_sources/semantic_network/SemGroups.txt
def get_semantic_types(category=None):
    # read in semantic types
    # st_df = pd.read_csv("https://semanticnetwork.nlm.nih.gov/download/SemGroups.txt", delimiter="|",
    #                     names=["x0", "x1", "x2", "x3"])
    # st_df = pd.read_csv("https://www.nlm.nih.gov/research/umls/knowledge_sources/semantic_network/SemGroups.txt", delimiter="|",
    #                     names=["x0", "x1", "x2", "x3"])
    st_df = pd.read_csv("./input/SemGroups.txt", delimiter="|",
                        names=["x0", "x1", "x2", "x3"])
    #st_d = dict(zip(st_df['x2'], st_df['x3']))
    # cat_df = st_df.query("x1 == @category")[['x2', 'x3']]
    # semantic_types = set(cat_df['x2'])
    # disorders = {'T037', 'T049', 'T048', 'T050', 'T184', 'T019', 'T190', 'T020', 'T033', 'T046', 'T191', 'T047'}
    semantic_types = st_df
    print(f'Read {st_df.shape[0]} rows')
    return semantic_types

def parse_descriptor_records(mesh_desc_path, semantic_types = None):
    """
    semantic_types is optional

    :param mesh_desc_path:
    :return:
    """
    #TODO: only care about disease related attributes. !!!
    # for example, ignoring RN CAS REGISTRY/EC NUMBER/UNII CODE  !!!

    attributes = {'MH': "term",
                  'MN': "tree",
                  'FX': "see_also",
                  'ST': "semantic_type",  # see: https://semanticnetwork.nlm.nih.gov/download/SemGroups.txt
                  'MS': "note",
                  'MR': "last_updated",
                  'DC': "descriptor_class",
                  'UI': "_id",
                  'RECTYPE': "record_type",
                  'synonyms': "synonyms"}  # added by me from PRINT ENTRY & ENTRY

    # TODO: parse PRINT ENTRY and ENTRY completely
    # "'D-2-hydroxyglutaric aciduria|T047|EQV|OMIM (2013)|ORD (2010)|090615|abdeef'"


    # read in the mesh data
    with open(mesh_desc_path) as f:
        mesh_desc = [x.strip() for x in f.readlines()]

    # which attributes can have multiple values?
    gb = filterfalse(lambda x: x[0], groupby(mesh_desc, lambda x: x == "*NEWRECORD"))
    ds = []
    for gb_record in gb:
        record = list(gb_record[1])
        d = dict(Counter([line.split("=", 1)[0].strip() for line in record if "=" in line]))
        ds.append(d)
    df = pd.DataFrame(ds).fillna(0)
    list_attribs = set(df.columns[df.max() > 1])
    # list_attribs = {'EC', 'ENTRY', 'FX', 'MH_TH', 'MN', 'PA', 'PI', 'PRINT ENTRY', 'RR', 'ST'}

    # split into records
    gb = filterfalse(lambda x: x[0], groupby(mesh_desc, lambda x: x == "*NEWRECORD"))

    mesh_terms = dict()
    for gb_record in gb:
        record = list(gb_record[1])
        d = defaultdict(list)
        for line in record:
            if "=" not in line:
                continue
            key = line.split("=", 1)[0].strip()
            value = line.split("=", 1)[1].strip()
            if key not in attributes and key not in list_attribs:
                d[key] = value
            elif key in list_attribs and key in attributes:
                d[attributes[key]].append(value)
            elif key in attributes and key not in {"PRINT ENTRY", "ENTRY"}:
                d[attributes[key]] = value
            elif key in {"PRINT ENTRY", "ENTRY"}:
                d['synonyms'].append(value.split("|", 1)[0])

        if semantic_types and not (set(d['semantic_type']) & semantic_types):
            continue
        mesh_terms[d['_id']] = dict(d)
    print(f'Completed. Read {len(mesh_terms)} MeSH Records.')
    return mesh_terms


def rank_terms(in_df, mh_df, filt):
    # Read input data to DataFrame
    df_1 = pd.read_csv(in_df, sep='|', names=['pmid', 'dp', 'ti', 'mh']) \
         .assign(mh=lambda df: df['mh'].str.split(';')) \
         .explode('mh', ignore_index=True) \
         .loc[:, ['pmid', 'mh']]
    
    df_4 = pd.DataFrame(filt.items(), columns=['sty', 'sty_name'])

    # Prepare MH DataFrame
    df_5 = mh_df.explode('sty', ignore_index=True) \
                .merge(right=df_4, how='right', on='sty') \
                .filter(items=['mh']) \
                .drop_duplicates()

    # Filter PMIDs by MHs corresponding to predefined semantic types
    df_6 = df_1.merge(right=df_5, how='inner', on='mh')

    # Ranking
    l2l = df_6.groupby('pmid')['mh'].apply(list).to_list()
    # From l2l_sel to TFIDF
    tf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None, min_df=3)
    tf_fit = tf.fit_transform(l2l)
    wrd_lst = tf.get_feature_names_out()
    score_lst = np.array(tf_fit.sum(axis=0)).reshape(-1).tolist()
    wrd2score = dict(zip(wrd_lst, score_lst))
    df_7 = pd.DataFrame().from_dict(wrd2score, orient='index') \
                         .reset_index() \
                         .set_axis(['name', 'score'], axis=1) \
                         .sort_values(by='score', ascending=False)

    return df_7