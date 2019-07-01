import os
import sys

import pandas as pd

from keras import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine import Model
from keras.layers import Embedding, GRU, merge, RepeatVector, TimeDistributed, Dense, Flatten
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from data_helpers import input_n_output_text_encoding_generator


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def read_content_only_file(file_path):

    with open(file_path) as f:
        x = f.read()
        reviews = x.split('\n')
    reviews = reviews[1: -1]  # Since 1st row would be taken up be header and last is blank

    return reviews


def build_delete_only_nn(seq_length, embedding_dim, vocab_size):

    # Define input placeholders for both text and attribute
    text_input = Input(shape=(seq_length,), name='text_input')
    attribute_input = Input(shape=(1,), name='attribute_input')

    # Encode content
    embedding_text = Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=seq_length)(text_input)
    gru_0 = GRU(512, return_sequences=False, name='gru_0')(embedding_text)

    # Encode attribute variable
    embedding_attribute = Embedding(output_dim=2, input_dim=2, input_length=1)(attribute_input)
    embedding_attribute = Flatten()(embedding_attribute)

    # Concatenate the embedding and encoded content string.
    merged_tensor = merge([gru_0, embedding_attribute], mode='concat', concat_axis=-1, name='merge')

    # Add decoder part
    repeat_merged_tensor = RepeatVector(seq_length)(merged_tensor)
    gru_1 = GRU(512, return_sequences=True, name='gru_1')(repeat_merged_tensor)
    time_dist_dense = TimeDistributed(Dense(vocab_size, activation='softmax'))(gru_1)

    # Put together the input and output layers
    final_model = Model(inputs=[text_input, attribute_input], outputs=[time_dist_dense])

    return final_model



if __name__ == '__main__':
    # Sys arguments
    dataset_name = sys.argv[1]  # Yelp, Amazon or Captions

    # Constants
    text_embed_dim = 128
    batch_size = 256
    random_state = 3
    no_epochs = 100
    epochs_patience_before_stopping = 7

    # File and directory paths
    par_data_dir = os.path.join('../data/sentiment_transfer_data', dataset_name)
    train_pos_file = os.path.join(par_data_dir, 'sentiment.train.1')
    train_neg_file = os.path.join(par_data_dir, 'sentiment.train.0')
    train_pos_content_file = os.path.join(par_data_dir, 'sentiment.train.content.1')
    train_neg_content_file = os.path.join(par_data_dir, 'sentiment.train.content.0')
    model_file_path = os.path.join(par_data_dir, 'style_transfer_model.hdf5')

    # Read the reviews files
    train_pos_df = pd.read_csv(train_pos_file, sep='\n', header=None)
    train_neg_df = pd.read_csv(train_neg_file, sep='\n', header=None)
    train_pos_reviews = list(train_pos_df[0])
    train_neg_reviews = list(train_neg_df[0])
    complete_reviews_list = train_pos_reviews + train_neg_reviews
    attribute_labels = [1] * len(train_pos_reviews) + [0] * len(train_neg_reviews)
    print ("Complete reviews list", len(complete_reviews_list))

    # Read the content only files
    train_pos_contents_list = read_content_only_file(train_pos_content_file)
    train_neg_contents_list = read_content_only_file(train_neg_content_file)
    complete_contents_list = train_pos_contents_list + train_neg_contents_list
    print ("Complete content only sentences list", len(complete_contents_list))

    # prepare reviews tokenizer
    reviews_tokenizer = create_tokenizer(complete_reviews_list)
    reviews_max_len = max([len(review.split()) for review in complete_reviews_list])
    reviews_vocab_size = len(reviews_tokenizer.word_index) + 1
    print ('Maximum len of review', reviews_max_len)
    print ('Reviews vocab size', reviews_vocab_size)

    # Split complete_reviews_list, complete_contents_list and attribute_labels into train and val set
    train_reviews_list, val_reviews_list, train_contents_list, val_contents_list, \
        train_attribute_labels, val_attribute_labels = train_test_split(
            complete_reviews_list, complete_contents_list, attribute_labels, shuffle=True, test_size=0.1,
            random_state=random_state
        )

    print ("Len of train reviews list", len(train_reviews_list))
    print ("Len of val reviews list", len(val_reviews_list))
    print ("Len of train contents list", len(train_contents_list))
    print ("Len of val contents list", len(val_contents_list))
    print ("Len of train attribute labels list", len(train_attribute_labels))
    print ("Len of val attribute labels list", len(val_attribute_labels))

    # Set up data generator for both training and validation data
    train_input_output_encoding_generator = input_n_output_text_encoding_generator(
        train_contents_list, train_reviews_list, train_attribute_labels, reviews_tokenizer,
        reviews_max_len, reviews_vocab_size, batch_size
    )
    val_input_output_encoding_generator = input_n_output_text_encoding_generator(
        val_contents_list, val_reviews_list, val_attribute_labels, reviews_tokenizer,
        reviews_max_len, reviews_vocab_size, batch_size
    )

    # # Test data generator
    # for X, Y in input_output_encoding_generator:
    #     print ("Input encoding shape", X[0].shape)
    #     print ("Attribute labels array shape", X[1].shape)
    #     print ("Output encoding shape", Y.shape)

    # Train the model
    model = build_delete_only_nn(reviews_max_len, text_embed_dim, reviews_vocab_size)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy')
    model.summary()

    # Model training begins
    steps_per_epoch = int(len(train_reviews_list) / batch_size)
    val_steps = int(len(val_reviews_list)/ batch_size)

    checkpointer = ModelCheckpoint(model_file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopper = EarlyStopping(monitor='val_loss', patience=epochs_patience_before_stopping)

    model.fit_generator(
        generator=train_input_output_encoding_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=no_epochs,
        validation_data=val_input_output_encoding_generator,
        validation_steps=val_steps,
        callbacks=[checkpointer, early_stopper]
    )
