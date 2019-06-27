"""This script is useful to create BERT features for a new model training set.
As this is a slow process it is separated from the training of the xgboost model.

The script only needs three inputs:

- data_input_path: Location of the data. Expects a csv file with a "Comment_text" and a "Hateful_or_not" column.
- data_output_path: Location where the BERT features will be saved.
    This file is used as input to "train_optimize_xgboost.py"
- bert_model_path: Location of the BERT model.
"""

import pandas as pd
import numpy as np

import tensorflow as tf

from bert_functions import *
from functions import *

# --------- User Input ----------
data_input_path = "data/APG-online-hate-classifier.csv"
data_output_path = "data/bert_features.csv"
bert_model_path = "bert_model.h5"
# -------------------------------

sess = tf.Session()
max_seq_length = 256
bert_model = build_model(max_seq_length, bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")
initialize_vars(sess)
bert_model.load_weights(bert_model_path)


X_train = pd.read_csv(data_input_path, sep=';')

# Create datasets (Only take up to max_seq_length words for memory)
train_text = X_train['Comment_text'].tolist()
train_text = [' '.join(str(t).split()[0:max_seq_length]) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = X_train['Hateful_or_not'].tolist()

# Instantiate tokenizer
tokenizer = create_tokenizer_from_hub_module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", sess)

# Convert data to InputExample format
train_examples = convert_text_to_examples(train_text, train_label)

# Convert to features
(train_input_ids, train_input_masks, train_segment_ids, train_labels
) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)

bert_train, bert_features = get_bert_features([train_input_ids, train_input_masks, train_segment_ids], bert_model)
X_train = X_train.reset_index().join(bert_train).set_index('index')

X_train.to_csv(data_output_path)