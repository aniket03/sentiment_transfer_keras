import os
import sys
import pickle

import numpy as np
import pandas as pd

from keras.models import load_model
from sklearn.utils import shuffle

from data_helpers import read_content_only_file, encode_sequences


def remove_repeated_words_from_seq(input_sentence):
    words_list = input_sentence.split()
    words_stack = []

    for word in words_list:
        # If last word entered in stack is same as present word in list then continue
        if len(words_stack) >=1 and word == words_stack[-1]:
            continue
        else:
            words_stack.append(word)

    final_string = ' '.join(words_stack)
    return final_string


if __name__ == '__main__':
    # Sys arguments
    dataset_name = sys.argv[1]  # Yelp, Amazon or Captions

    # File and directory paths
    # par_data_dir = os.path.join('/content/drive/My Drive/deep_learning_work', dataset_name)  # Used on Google colab
    par_data_dir = os.path.join('../data/sentiment_transfer_data', dataset_name)
    test_pos_file = os.path.join(par_data_dir, 'sentiment.test.1')
    test_neg_file = os.path.join(par_data_dir, 'sentiment.test.0')
    test_pos_content_file = os.path.join(par_data_dir, 'sentiment.test.content.1')
    test_neg_content_file = os.path.join(par_data_dir, 'sentiment.test.content.0')
    model_file_path = os.path.join(par_data_dir, 'style_transfer_model.hdf5')
    tokenizer_file_path = os.path.join(par_data_dir, 'reviews_tokenizer.pkl')
    max_seq_len = 15
    random_state = 3

    # Read the reviews files
    test_pos_df = pd.read_csv(test_pos_file, sep='\n', header=None)
    test_neg_df = pd.read_csv(test_neg_file, sep='\n', header=None)
    test_pos_reviews = list(test_pos_df[0])
    test_neg_reviews = list(test_neg_df[0])
    complete_reviews_list = test_pos_reviews + test_neg_reviews
    attribute_labels = [1] * len(test_pos_reviews) + [0] * len(test_neg_reviews)
    print ("Len of complete reviews list", len(complete_reviews_list))

    # Read the content only files
    test_pos_contents_list = read_content_only_file(test_pos_content_file)
    test_neg_contents_list = read_content_only_file(test_neg_content_file)
    complete_contents_list = test_pos_contents_list + test_neg_contents_list
    print ("Len of complete content only sentences list", len(complete_contents_list))

    # Reviews to test - final
    # Split complete_reviews_list, complete_contents_list and attribute_labels into train and val set
    test_reviews_list, test_contents_list, test_attribute_labels = shuffle(
        complete_reviews_list, complete_contents_list, attribute_labels, random_state=random_state
    )

    # Just for test purpose
    test_reviews_list = test_reviews_list[:30]
    test_contents_list = test_contents_list[:30]
    test_attribute_labels = test_attribute_labels[:30]
    test_attribute_labels = np.array(test_attribute_labels)

    # Revert the attribute labels to switch sentiment
    for ind in range(len(test_attribute_labels)):
        test_attribute_labels[ind] = 1 if test_attribute_labels[ind] == 0 else 0

    # Tokenize
    with open(tokenizer_file_path, 'rb') as fp:
        reviews_tokenizer = pickle.load(fp)
        reverse_word_map = dict(map(reversed, reviews_tokenizer.word_index.items()))

    test_sequences_arr = encode_sequences(reviews_tokenizer, max_seq_len, test_contents_list)
    print (test_sequences_arr.shape)

    # Get prediction from model
    text_gen_model = load_model(model_file_path)
    text_gen_probs = text_gen_model.predict(
        [test_sequences_arr, test_attribute_labels]
    )
    for test_review, test_content, text_gen_prob_mat in zip(test_reviews_list, test_contents_list, text_gen_probs):
        generated_seq = np.argmax(text_gen_prob_mat, axis=1)
        generated_word_seq = []
        for word_ind in generated_seq:
            try:
                generated_word_seq.append(reverse_word_map[word_ind])
            except KeyError:
                generated_word_seq.append('')
        generated_text = " ".join(generated_word_seq)
        processed_generated_text = remove_repeated_words_from_seq(generated_text)

        print ("\n")
        print ("Actual review: ", test_review)
        # print ("Content only", test_content)
        print ("Generated text: ", processed_generated_text)
