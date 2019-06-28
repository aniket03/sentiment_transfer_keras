import os
import re
import sys

import pandas as pd
from sklearn.utils import shuffle


def delete_attribute_content(input_text, list_attribute_markers):
    """
    Given input text and list_attribute_markers, finds the attribute_marker that is present in the input_text,
    replaces that with blank string, and returns the content words string, with the attribute markers present.
    """

    # Create a regex for attribute markers
    bounded_attribute_markers = ['\\b{}\\b'.format(attribute_marker) for attribute_marker in list_attribute_markers]
    orred_attribute_markers = '|'.join(bounded_attribute_markers)
    attribute_markers_re = r'%s' % orred_attribute_markers

    # Delete attribute words from the text
    content_words_text = re.sub(attribute_markers_re, '', input_text)
    attribute_markers_matched = re.findall(attribute_markers_re, input_text)

    return content_words_text, attribute_markers_matched


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
    train_pos_reviews = list(train_pos_df[0])
    train_neg_reviews = list(train_neg_df[0])

    # Read the attribute markers' file
    pos_attribute_markers_df = pd.read_csv(os.path.join(par_data_dir, 'pos_attribute_markers.csv'))
    neg_attribute_markers_df = pd.read_csv(os.path.join(par_data_dir, 'neg_attribute_markers.csv'))
    pos_attribute_markers = pos_attribute_markers_df['pos_attribute_markers']
    neg_attribute_markers = neg_attribute_markers_df['neg_attribute_markers']

    # Shuffle the reviews list,
    train_pos_reviews = shuffle(train_pos_reviews)
    train_neg_reviews = shuffle(train_neg_reviews)

    for pos_review in train_pos_reviews[:3]:
        content_phrase, attribute_phrases = delete_attribute_content(pos_review, pos_attribute_markers)
        print ("Review", pos_review)
        print ("Content", content_phrase)
        print ("Attribute phrases", attribute_phrases)

    for neg_review in train_neg_reviews[:3]:
        content_phrase, attribute_phrases = delete_attribute_content(neg_review, neg_attribute_markers)
        print ("Review", neg_review)
        print ("Content", content_phrase)
        print ("Attribute phrases", attribute_phrases)

