"""This can be used to predict the labels (hateful or not) of new data with existing models.

Before you run it, you have to change the following input variables;

- data_path: Location of the new data.
- bert_model_path: Location of the BERT model.
- xgb_model_path: Location of the predictions model.
- bert_output_path = Location where the BERT features should be saved.
- predictions_output_path = Location where the predictions are saved.

The data_path should point to a csv file with a column called "Comment_text".

BERT is a huge machine learning model and it takes quite long to generate features with it, that why the BERT
features are stored. If for some reason an error happens afterwards, the BERT features don't have to be computed again.

The output is a csv file with three columns: the original comment, the predicted probability that is is a hateful
comment and the binary prediction (True if the predicted probability is greater than 0.5).

To execute the script you just need to change the variables in the "User Input" section and then run the script.
"""


import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf

from sklearn.externals import joblib
from bert_functions import *
from functions import *

# --------- User Input ----------------------------
data_path = "data/new_data.csv"
bert_model_path = "bert_model.h5"
xgb_model_path = "xgboost_model.dat"
bert_output_path = "data/bert_output.csv"
predictions_output_path = "data/predictions.csv"
# -------------------------------------------------

new_data = pd.read_csv(data_path)

sess = tf.Session()
max_seq_length = 256
bert_model = build_model(max_seq_length, bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")
initialize_vars(sess)
bert_model.load_weights(bert_model_path)

# create datasets for BERT
text = new_data['Comment_text'].tolist()
text = [' '.join(str(t).split()[0:max_seq_length]) for t in text]
text = np.array(text, dtype=object)[:, np.newaxis]
pseudo_label = [1] * len(text)

tokenizer = create_tokenizer_from_hub_module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", sess)

# Convert data to InputExample format
examples = convert_text_to_examples(text, pseudo_label)

# Convert to features
(input_ids, input_masks, segment_ids, labels
) = convert_examples_to_features(tokenizer, examples, max_seq_length=max_seq_length)

bert_data, bert_features = get_bert_features([input_ids, input_masks, segment_ids], bert_model)
bert_data = new_data.reset_index().join(bert_data).set_index('index')
bert_data.to_csv(bert_output_path)

xgb_model = joblib.load(xgb_model_path)
bert_data = pd.read_csv(bert_output_path)
preds = xgb_model.predict(bert_data.loc[:, bert_features])
predictions = new_data
predictions['prediciton_score'] = preds
predictions['predictions'] = preds > 0.5

predictions.to_csv(predictions_output_path)

