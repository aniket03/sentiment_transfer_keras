import os
import pandas as pd

if __name__ == '__main__':
    # Constants and file paths
    par_data_dir = '../data/sentiment_transfer_data/yelp/'
    yelp_train_pos_file = os.path.join(par_data_dir, 'sentiment.train.1')
    yelp_train_neg_file = os.path.join(par_data_dir, 'sentiment.train.0')
    yelp_val_pos_file = os.path.join(par_data_dir, 'sentiment.dev.1')
    yelp_val_neg_file = os.path.join(par_data_dir, 'sentiment.dev.0')
    yelp_test_pos_file = os.path.join(par_data_dir, 'sentiment.test.1')
    yelp_test_neg_file = os.path.join(par_data_dir, 'sentiment.test.0')

    # Read the positive reviews files
    yelp_train_pos_df = pd.read_csv(yelp_train_pos_file, sep='\n', header=None)
    yelp_val_pos_df = pd.read_csv(yelp_val_pos_file, sep='\n', header=None)
    yelp_test_pos_df = pd.read_csv(yelp_test_pos_file, sep='\n', header=None)

    # Read the negative reviews files
    yelp_train_neg_df = pd.read_csv(yelp_train_neg_file, sep='\n', header=None)
    yelp_val_neg_df = pd.read_csv(yelp_val_neg_file, sep='\n', header=None)
    yelp_test_neg_df = pd.read_csv(yelp_test_neg_file, sep='\n', header=None)

    # Print no of samples in each file
    print ("No of Yelp pos training samples", len(yelp_train_pos_df))
    print ("No of Yelp neg training samples", len(yelp_train_neg_df))
    print ("No of Yelp pos validation samples", len(yelp_val_pos_df))
    print ("No of Yelp neg validation samples", len(yelp_val_neg_df))
    print ("No of Yelp pos test samples", len(yelp_test_pos_df))
    print ("No of Yelp neg test samples", len(yelp_test_neg_df))
