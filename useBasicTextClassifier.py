# This code uses the Basic Text Classifer trained using the Keras API

import matplotlib.pyplot as plt
import re
import string
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
import numpy as np 


#We will need to declare the
#function to do preprocessing to remove html tags from
# in the same way we did in the training phase  
@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')
#We will create a standardization layer to 
# tandardize, tokenize, and vectorize our data
max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

new_model = tf.keras.models.load_model('my_model')
new_model.summary()

examples = np.array([
  "The movie was great!",
  "Overal nice movie but some parts were terrible",
  "The movie was terrible..."
])
# print(new_model)
print(new_model.predict(examples))

# Save the entire model as a SavedModel


