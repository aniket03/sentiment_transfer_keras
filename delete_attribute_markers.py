import os
import re
import sys

import pandas as pd
from sklearn.utils import shuffle


def build_attribute_markers_regex(list_attribute_markers):
    """
    Create a regex for attribute markers list
    :param list_attribute_markers: List of attribute markers to spot and replace
    :return: attribute_markers_re
    """

    bounded_attribute_markers = ['\\b{}\\b'.format(attribute_marker) for attribute_marker in list_attribute_markers]
    orred_attribute_markers = '|'.join(bounded_attribute_markers)
    attribute_markers_re = r'%s' % orred_attribute_markers

    return attribute_markers_re


def delete_attribute_content(input_text, attribute_markers_re):
    """
    Given input text and attribute_markers_re, finds the attribute_marker that is present in the input_text,
    replaces that with blank string, and returns the content words string, with the attribute markers present.
    """

    content_words_text = re.sub(attribute_markers_re, '', input_text)
    # attribute_markers_matched = re.findall(attribute_markers_re, input_text)

    return content_words_text


if __name__ == '__main__':

    # Sys arguments
    dataset_name = sys.argv[1]  # Yelp, Amazon or Captions

    # Constants and variables
    par_data_dir = os.path.join('../data/sentiment_transfer_data', dataset_name)
    train_pos_file = os.path.join(par_data_dir, 'sentiment.train.1')
    train_neg_file = os.path.join(par_data_dir, 'sentiment.train.0')
    pos_attribute_markers_file = os.path.join(par_data_dir, 'pos_attribute_markers.csv')
    neg_attribute_markers_file = os.path.join(par_data_dir, 'neg_attribute_markers.csv')
    train_pos_content_file = os.path.join(par_data_dir, 'sentiment.train.content.1')
    train_neg_content_file = os.path.join(par_data_dir, 'sentiment.train.content.0')

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

    # Build regexes for pos and neg attribute markers lists
    pos_attribute_markers_re = build_attribute_markers_regex(pos_attribute_markers)
    print ("Pos attribute marker regex built")
    neg_attribute_markers_re = build_attribute_markers_regex(neg_attribute_markers)
    print ("Neg attribute marker regex built")

    # Delete attribute content from positive reviews
    pos_reviews_content_list = []
    for pos_review in train_pos_reviews:
        content_phrase = delete_attribute_content(pos_review, pos_attribute_markers_re)
        pos_reviews_content_list.append(content_phrase)

    # Write pos reviews content only part to file
    pos_content_df = pd.DataFrame()
    pos_content_df['only_content_part'] = pos_reviews_content_list
    pos_content_df.to_csv(train_pos_content_file, index=False)

    print ("Pos reviews content and attribute markers separated")

    # Delete attribute content from negative reviews
    neg_reviews_content_list = []
    for neg_review in train_neg_reviews:
        content_phrase = delete_attribute_content(neg_review, neg_attribute_markers_re)
        neg_reviews_content_list.append(content_phrase)

    # Write neg reviews content only part to file
    neg_content_df = pd.DataFrame()
    neg_content_df['only_content_part'] = neg_reviews_content_list
    neg_content_df.to_csv(train_neg_content_file, index=False)

    print ("Neg reviews content and attribute markers separated")
