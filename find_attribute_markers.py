import os
import sys

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


def get_attribute_markers(pos_reviews_list, neg_reviews_list):

    # Constants
    lmda = 1
    threshold_for_attribute_marker = 15

    # Define count vectorizers
    count_vect_pos = CountVectorizer(ngram_range=(1, 4))
    count_vect_neg = CountVectorizer(ngram_range=(1, 4))

    # Get counts sparse matrices for pos and neg reviews
    pos_phrases_counts_sp_mat = count_vect_pos.fit_transform(pos_reviews_list)
    neg_phrases_counts_sp_mat = count_vect_neg.fit_transform(neg_reviews_list)

    # Get positive vocabulary and negative vocabulary
    pos_phrases = count_vect_pos.vocabulary_
    neg_phrases = count_vect_neg.vocabulary_
    all_phrases = list(set(pos_phrases) | set(neg_phrases))

    # Sum the counts of pos and neg phrases' counts across reviews
    pos_phrases_counts = np.squeeze(np.array(np.sum(pos_phrases_counts_sp_mat, axis=0)))
    neg_phrases_counts = np.squeeze(np.array(np.sum(neg_phrases_counts_sp_mat, axis=0)))

    # Make dict mapping between phrase and their counts
    pos_phrases_counts_dict = dict(zip(pos_phrases, pos_phrases_counts))
    neg_phrases_counts_dict = dict(zip(neg_phrases, neg_phrases_counts))

    # Get attribute markers
    pos_attribute_markers = []
    neg_attribute_markers = []

    for phrase in all_phrases:
        try:
            pos_count = pos_phrases_counts_dict[phrase]
        except KeyError:
            pos_count = 0

        try:
            neg_count = neg_phrases_counts_dict[phrase]
        except KeyError:
            neg_count = 0

        pos_salience = (pos_count + lmda) / (neg_count + lmda)
        neg_salience = (neg_count + lmda) / (pos_count + lmda)

        if pos_salience >= threshold_for_attribute_marker:
            pos_attribute_markers.append(phrase)
        if neg_salience >= threshold_for_attribute_marker:
            neg_attribute_markers.append(phrase)

    return pos_attribute_markers, neg_attribute_markers


if __name__ == '__main__':
    # Sys arguments
    dataset_name = sys.argv[1]  # Yelp, Amazon or Captions

    # Constants and variables
    par_data_dir = os.path.join('../data/sentiment_transfer_data', dataset_name)
    train_pos_file = os.path.join(par_data_dir, 'sentiment.train.1')
    train_neg_file = os.path.join(par_data_dir, 'sentiment.train.0')
    pos_attribute_markers_file = os.path.join(par_data_dir, 'pos_attribute_markers.csv')
    neg_attribute_markers_file = os.path.join(par_data_dir, 'neg_attribute_markers.csv')

    # Read the reviews files
    train_pos_df = pd.read_csv(train_pos_file, sep='\n', header=None)
    train_neg_df = pd.read_csv(train_neg_file, sep='\n', header=None)

    # Get attribute markers
    train_pos_reviews = list(train_pos_df[0])
    train_neg_reviews = list(train_neg_df[0])
    pos_attribute_markers, neg_attribute_markers = get_attribute_markers(train_pos_reviews, train_neg_reviews)

    # Save the attribute markers
    pos_attribute_markers_df = pd.DataFrame()
    pos_attribute_markers_df['pos_attribute_markers'] = pos_attribute_markers
    neg_attribute_markers_df = pd.DataFrame()
    neg_attribute_markers_df['neg_attribute_markers'] = neg_attribute_markers
    pos_attribute_markers_df.to_csv(pos_attribute_markers_file)
    neg_attribute_markers_df.to_csv(neg_attribute_markers_file)
