import os
import re
import sys

import pandas as pd


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


def delete_attribute_content(input_text_list, attribute_markers_re):
    """
    Given input text_list and attribute_markers_re, finds the attribute_marker that is present in each input_text,
    replaces that with blank string, and returns the content words strings list.
    """
    content_only_list = []
    for input_text in input_text_list:
        content_words_text = re.sub(attribute_markers_re, '', input_text)
        content_only_list.append(content_words_text)

    return content_only_list


def write_content_only_sentences_file(content_only_list, file_path):
    """
    Writes the content_only_sentences list to the appropriate file path
    :param content_only_list: List of content only sentences
    :param file_path: file_path to write to
    """
    content_df = pd.DataFrame()
    content_df['only_content_part'] = content_only_list
    content_df.to_csv(file_path, index=False)


if __name__ == '__main__':

    # Sys arguments
    dataset_name = sys.argv[1]  # Yelp, Amazon or Captions

    # Constants and variables
    par_data_dir = os.path.join('../data/sentiment_transfer_data', dataset_name)
    train_pos_file = os.path.join(par_data_dir, 'sentiment.train.1')
    train_neg_file = os.path.join(par_data_dir, 'sentiment.train.0')
    val_pos_file = os.path.join(par_data_dir, 'sentiment.dev.1')
    val_neg_file = os.path.join(par_data_dir, 'sentiment.dev.0')
    test_pos_file = os.path.join(par_data_dir, 'sentiment.test.1')
    test_neg_file = os.path.join(par_data_dir, 'sentiment.test.0')
    pos_attribute_markers_file = os.path.join(par_data_dir, 'pos_attribute_markers.csv')
    neg_attribute_markers_file = os.path.join(par_data_dir, 'neg_attribute_markers.csv')
    train_pos_content_file = os.path.join(par_data_dir, 'sentiment.train.content.1')
    train_neg_content_file = os.path.join(par_data_dir, 'sentiment.train.content.0')
    val_pos_content_file = os.path.join(par_data_dir, 'sentiment.dev.content.1')
    val_neg_content_file = os.path.join(par_data_dir, 'sentiment.dev.content.0')
    test_pos_content_file = os.path.join(par_data_dir, 'sentiment.test.content.1')
    test_neg_content_file = os.path.join(par_data_dir, 'sentiment.test.content.0')

    # Read the reviews files
    train_pos_reviews = pd.read_csv(train_pos_file, sep='\n', header=None)[0]
    train_neg_reviews = pd.read_csv(train_neg_file, sep='\n', header=None)[0]
    val_pos_reviews = pd.read_csv(val_pos_file, sep='\n', header=None)[0]
    val_neg_reviews = pd.read_csv(val_neg_file, sep='\n', header=None)[0]
    test_pos_reviews = pd.read_csv(test_pos_file, sep='\n', header=None)[0]
    test_neg_reviews = pd.read_csv(test_neg_file, sep='\n', header=None)[0]

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
    train_pos_reviews_content_list = delete_attribute_content(train_pos_reviews, pos_attribute_markers_re)
    write_content_only_sentences_file(train_pos_reviews_content_list, train_pos_content_file)
    val_pos_reviews_content_list = delete_attribute_content(val_pos_reviews, pos_attribute_markers_re)
    write_content_only_sentences_file(val_pos_reviews_content_list, val_pos_content_file)
    test_pos_reviews_content_list = delete_attribute_content(test_pos_reviews, pos_attribute_markers_re)
    write_content_only_sentences_file(test_pos_reviews_content_list, test_pos_content_file)

    # Delete attribute content from negative reviews
    train_neg_reviews_content_list = delete_attribute_content(train_neg_reviews, neg_attribute_markers_re)
    write_content_only_sentences_file(train_neg_reviews_content_list, train_neg_content_file)
    val_neg_reviews_content_list = delete_attribute_content(val_neg_reviews, neg_attribute_markers_re)
    write_content_only_sentences_file(val_neg_reviews_content_list, val_neg_content_file)
    test_neg_reviews_content_list = delete_attribute_content(test_neg_reviews, neg_attribute_markers_re)
    write_content_only_sentences_file(test_neg_reviews_content_list, test_neg_content_file)
