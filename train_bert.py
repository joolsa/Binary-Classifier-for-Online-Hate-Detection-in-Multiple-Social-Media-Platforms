""" This script fine-tunes the BERT model to the given dataset.
You have to provide two variables:

- data_path: Location of the data. Expects a csv file with a "Comment_text" and a "Hateful_or_not" column.
- model_name = Location where the model will be saved.

Finetuning a gigantic model like BERT takes a lot of time. It makes sense to use cloud computing
resources to run this script.
Make sure to exclude the test set from the BERT finetuning process, otherwise you'll have data leakage.
"""

import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split
from bert_functions import *


# --------- User Input ----------------------------
data_path = "data/APG-online-hate-classifier.csv"
model_name = "bert_model"
# -------------------------------------------------


data = pd.read_csv(data_path, sep=';')
X_train, X_test = train_test_split(data, test_size=0.25)
print(X_train.shape)
print(X_test.shape)

# Initialize session
sess = tf.Session()

# Params for bert model and tokenization
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
# max comment length for padding
max_seq_length = 256


# Create datasets (Only take up to max_seq_length words for memory)
train_text = X_train['Comment_text'].tolist()
train_text = [' '.join(str(t).split()[0:max_seq_length]) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = X_train['Hateful_or_not'].tolist()

test_text = X_test['Comment_text'].tolist()
test_text = [' '.join(t.split()[0:max_seq_length]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = X_test['Hateful_or_not'].tolist()


# Instantiate tokenizer
tokenizer = create_tokenizer_from_hub_module(bert_path, sess)

# Convert data to InputExample format
train_examples = convert_text_to_examples(train_text, train_label)
test_examples = convert_text_to_examples(test_text, test_label)

# Convert to features
(train_input_ids, train_input_masks, train_segment_ids, train_labels
) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)
(test_input_ids, test_input_masks, test_segment_ids, test_labels
) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)



bert_model = build_model(max_seq_length, bert_path=bert_path)

# Instantiate variables
initialize_vars(sess)

bert_model.fit(
    [train_input_ids, train_input_masks, train_segment_ids],
    train_labels,
    validation_data=([test_input_ids, test_input_masks, test_segment_ids], test_labels),
    epochs=1,
    batch_size=128
)

bert_model.save(model_name + ".h5")