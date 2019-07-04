import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def read_content_only_file(file_path):

    with open(file_path) as f:
        x = f.read()
        reviews = x.split('\n')
    reviews = reviews[1: -1]  # Since 1st row would be taken up be header and last is blank

    return reviews


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


def input_n_output_text_encoding_generator(input_texts_list, output_texts_list,
                                           labels_list, text_tokenizer, seq_len,
                                           vocab_size, batch_size):

    """
    Generates both input and output text encoding for passing as X and Y to the text generation model
    :param input_texts_list: List of input texts
    :param output_texts_list: List of output texts
    :param labels_list: List of labels where labels_list[i] => the label that is associated with output_texts_list[i]
    :param text_tokenizer: Tokenizer object
    :param seq_len: The length each sequence should be of.
    :param vocab_size: The size of vocab that was used when fitting tokenizer
    :param batch_size: Batch size in which data is to be generated
    :return:
    """

    text_index = 0
    total_texts_present = len(input_texts_list)

    while True:

        if text_index <= total_texts_present - batch_size:
            input_text_batch = input_texts_list[text_index: text_index + batch_size]
            output_text_batch = output_texts_list[text_index: text_index + batch_size]
            labels_batch = labels_list[text_index: text_index + batch_size]

            if text_index < total_texts_present - batch_size:
                text_index += batch_size
            else:
                text_index = 0
        else:
            texts_covered_till_end = total_texts_present - text_index
            input_text_batch = input_texts_list[text_index:] + \
                               input_texts_list[0: batch_size - texts_covered_till_end]
            output_text_batch = output_texts_list[text_index:] + \
                                output_texts_list[0: batch_size - texts_covered_till_end]
            labels_batch = labels_list[text_index:] +\
                           labels_list[0: batch_size - texts_covered_till_end]
            text_index = batch_size - texts_covered_till_end

        encoded_input_batch = encode_sequences(text_tokenizer, seq_len, input_text_batch)
        encoded_output_batch = encode_sequences(text_tokenizer, seq_len, output_text_batch)
        one_hot_output_batch = encode_output(encoded_output_batch, vocab_size)
        labels_batch = np.array(labels_batch)

        yield [encoded_input_batch, labels_batch], one_hot_output_batch
