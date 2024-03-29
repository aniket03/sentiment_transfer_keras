import os
import sys
import pickle

import pandas as pd

from keras import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.engine import Model
from keras.layers import Embedding, GRU, merge, RepeatVector, TimeDistributed, Dense, Flatten
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from sklearn.utils import shuffle

from data_helpers import input_n_output_text_encoding_generator, read_content_only_file


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


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
    initial_lr = 1e-2
    epochs_patience_before_decay = 2
    epochs_patience_before_stopping = 5

    # File and directory paths
    # par_data_dir = os.path.join('/content/drive/My Drive/deep_learning_work', dataset_name)  # Used on Google colab
    par_data_dir = os.path.join('../data/sentiment_transfer_data', dataset_name)
    train_pos_file = os.path.join(par_data_dir, 'sentiment.train.1')
    train_neg_file = os.path.join(par_data_dir, 'sentiment.train.0')
    val_pos_file = os.path.join(par_data_dir, 'sentiment.dev.1')
    val_neg_file = os.path.join(par_data_dir, 'sentiment.dev.0')
    train_pos_content_file = os.path.join(par_data_dir, 'sentiment.train.content.1')
    train_neg_content_file = os.path.join(par_data_dir, 'sentiment.train.content.0')
    val_pos_content_file = os.path.join(par_data_dir, 'sentiment.dev.content.1')
    val_neg_content_file = os.path.join(par_data_dir, 'sentiment.dev.content.0')
    model_file_path = os.path.join(par_data_dir, 'style_transfer_model.hdf5')
    tokenizer_file_path = os.path.join(par_data_dir, 'reviews_tokenizer.pkl')

    # Read the reviews files
    train_pos_reviews = list(pd.read_csv(train_pos_file, sep='\n', header=None)[0])
    train_neg_reviews = list(pd.read_csv(train_neg_file, sep='\n', header=None)[0])
    val_pos_reviews = list(pd.read_csv(val_pos_file, sep='\n', header=None)[0])
    val_neg_reviews = list(pd.read_csv(val_neg_file, sep='\n', header=None)[0])

    # Concatenate separate train-pos train-neg and val-pos val-neg
    train_reviews_list = train_pos_reviews + train_neg_reviews
    val_reviews_list = val_pos_reviews + val_neg_reviews
    complete_reviews_list = train_reviews_list + val_reviews_list
    train_attribute_labels = [1] * len(train_pos_reviews) + [0] * len(train_neg_reviews)
    val_attribute_labels = [1] * len(val_pos_reviews) + [0] * len(val_neg_reviews)
    print ("Complete train reviews list", len(train_reviews_list))
    print ("Complete val reviews list", len(val_reviews_list))

    # Read the content only files
    train_pos_contents_list = read_content_only_file(train_pos_content_file)
    train_neg_contents_list = read_content_only_file(train_neg_content_file)
    val_pos_contents_list = read_content_only_file(val_pos_content_file)
    val_neg_contents_list = read_content_only_file(val_neg_content_file)
    train_contents_list = train_pos_contents_list + train_neg_contents_list
    val_contents_list = val_pos_contents_list + val_neg_contents_list
    print ("Complete train content only sentences list", len(train_contents_list))
    print ("Complete val content only sentences list", len(val_contents_list))

    # prepare reviews tokenizer
    reviews_tokenizer = create_tokenizer(complete_reviews_list)
    reviews_max_len = max([len(review.split()) for review in complete_reviews_list])
    reviews_vocab_size = len(reviews_tokenizer.word_index) + 1
    print ('Maximum len of review', reviews_max_len)
    print ('Reviews vocab size', reviews_vocab_size)
    with open(tokenizer_file_path, 'wb') as fp:
        pickle.dump(reviews_tokenizer, fp, protocol=4)

    # Shuffle {train/ val}_reviews_list, {train/ val}_contents_list and {train/ val}_attribute_labels
    train_reviews_list, train_contents_list, train_attribute_labels = shuffle(
        train_reviews_list, train_contents_list, train_attribute_labels, random_state=random_state
    )
    val_reviews_list, val_contents_list, val_attribute_labels = shuffle(
        val_reviews_list, val_contents_list, val_attribute_labels, random_state=random_state
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

    # Compile the model
    if os.path.exists(model_file_path):
        model = load_model(model_file_path)
    else:
        model = build_delete_only_nn(reviews_max_len, text_embed_dim, reviews_vocab_size)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy')
    model.summary()

    # Training and validation steps
    steps_per_epoch = int(len(train_reviews_list) / batch_size)
    val_steps = int(len(val_reviews_list)/ batch_size)

    # Define model training call backs
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=epochs_patience_before_decay,
                                             verbose=1, min_lr=1e-7)
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
