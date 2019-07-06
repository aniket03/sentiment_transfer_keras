import os
import sys

import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer


def eliminate_stop_words(list_reviews):
    stop_words = [
        'is', 'are', 'was', 'were', 'and', 'be', 'could', 'did', 'do', 'does', 'even', 'had', 'has', 'have', 'in',
        'or', 'he', 'she', 'his', 'her', 'the', 'their', 'them', 'they', 'this', 'to', 'went', '_num_', 'i', 'you',
        'am', 'will', 'would', 'your', 'my', 'we', 'there', 'these'
    ]

    bounded_stop_words = ['\\b{}\\b'.format(stop_word) for stop_word in stop_words]
    orred_stop_words = '|'.join(bounded_stop_words)
    orred_stop_words_re = r'%s' % orred_stop_words

    stop_words_removed_list_reviews = [re.sub(orred_stop_words_re, '', review) for review in list_reviews]

    return stop_words_removed_list_reviews


def get_attribute_markers(pos_reviews_list, neg_reviews_list):

    # Constants
    lmda = 1
    threshold_for_attribute_marker = 15

    # Define count vectorizers
    count_vect_pos = CountVectorizer(ngram_range=(1, 2), binary=True)
    count_vect_neg = CountVectorizer(ngram_range=(1, 2), binary=True)

    # Get counts sparse matrices for pos and neg reviews
    pos_phrases_counts_sp_mat = count_vect_pos.fit_transform(pos_reviews_list)
    neg_phrases_counts_sp_mat = count_vect_neg.fit_transform(neg_reviews_list)

    # Get positive vocabulary and negative vocabulary
    pos_phrases_indices_map = count_vect_pos.vocabulary_
    neg_phrases_indices_map = count_vect_neg.vocabulary_

    # Define set of all_phrases
    all_phrases = list(set(pos_phrases_indices_map.keys()) | set(neg_phrases_indices_map.keys()))

    # Sum the counts of pos and neg phrases' counts across reviews
    pos_phrases_counts = np.squeeze(np.array(np.sum(pos_phrases_counts_sp_mat, axis=0)))
    neg_phrases_counts = np.squeeze(np.array(np.sum(neg_phrases_counts_sp_mat, axis=0)))

    # Get attribute markers
    pos_attribute_markers = []
    neg_attribute_markers = []

    for phrase in all_phrases:
        try:
            pos_count = pos_phrases_counts[pos_phrases_indices_map[phrase]]
        except KeyError:
            pos_count = 0

        try:
            neg_count = neg_phrases_counts[neg_phrases_indices_map[phrase]]
        except KeyError:
            neg_count = 0

        pos_salience = (pos_count + lmda) / (neg_count + lmda)
        neg_salience = (neg_count + lmda) / (pos_count + lmda)

        if pos_salience >= threshold_for_attribute_marker:
            pos_attribute_markers.append(phrase)
        if neg_salience >= threshold_for_attribute_marker:
            neg_attribute_markers.append(phrase)

    return pos_attribute_markers, neg_attribute_markers


def sort_attribute_markers_by_len(attibute_markers_list):
    """
    Sort attribute markers basis their length
    """
    attribute_markers_with_len = [[len(attribute_marker.split()), attribute_marker]
                                  for attribute_marker in attibute_markers_list]
    attribute_markers_with_len.sort()

    attribute_markers_list = [attribute_marker_with_len[1] for attribute_marker_with_len in attribute_markers_with_len]

    return attribute_markers_list


if __name__ == '__main__':
    # Sys arguments
    dataset_name = sys.argv[1]  # Yelp, Amazon or Captions

    # Constants and variables
    par_data_dir = os.path.join('../data/sentiment_transfer_data', dataset_name)
    train_pos_file = os.path.join(par_data_dir, 'sentiment.train.1')
    train_neg_file = os.path.join(par_data_dir, 'sentiment.train.0')
    val_pos_file = os.path.join(par_data_dir, 'sentiment.dev.1')
    val_neg_file = os.path.join(par_data_dir, 'sentiment.dev.0')
    pos_attribute_markers_file = os.path.join(par_data_dir, 'pos_attribute_markers.csv')
    neg_attribute_markers_file = os.path.join(par_data_dir, 'neg_attribute_markers.csv')

    # Read the reviews files
    train_pos_df = pd.read_csv(train_pos_file, sep='\n', header=None)
    train_neg_df = pd.read_csv(train_neg_file, sep='\n', header=None)
    val_pos_df = pd.read_csv(val_pos_file, sep='\n', header=None)
    val_neg_df = pd.read_csv(val_neg_file, sep='\n', header=None)

    # Get list of pos and neg reviews
    train_pos_reviews = list(train_pos_df[0])
    train_neg_reviews = list(train_neg_df[0])
    val_pos_reviews = list(val_pos_df[0])
    val_neg_reviews = list(val_neg_df[0])

    # Remove stop words
    train_pos_reviews = eliminate_stop_words(train_pos_reviews)
    train_neg_reviews = eliminate_stop_words(train_neg_reviews)
    val_pos_reviews = eliminate_stop_words(val_pos_reviews)
    val_neg_reviews = eliminate_stop_words(val_neg_reviews)

    # Combine train and val reviews
    pos_reviews = train_pos_reviews + val_pos_reviews
    neg_reviews = train_neg_reviews + val_neg_reviews

    # Get attribute markers
    pos_attribute_markers, neg_attribute_markers = get_attribute_markers(pos_reviews, neg_reviews)

    # Sort pos and neg attribute markers basis their length
    pos_attribute_markers = sort_attribute_markers_by_len(pos_attribute_markers)
    neg_attribute_markers = sort_attribute_markers_by_len(neg_attribute_markers)

    # Save the attribute markers
    pos_attribute_markers_df = pd.DataFrame()
    pos_attribute_markers_df['pos_attribute_markers'] = pos_attribute_markers
    neg_attribute_markers_df = pd.DataFrame()
    neg_attribute_markers_df['neg_attribute_markers'] = neg_attribute_markers
    pos_attribute_markers_df.to_csv(pos_attribute_markers_file)
    neg_attribute_markers_df.to_csv(neg_attribute_markers_file)
