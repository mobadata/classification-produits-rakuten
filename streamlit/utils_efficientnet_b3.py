import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import f1_score
from prdtypecode_labels import prdtypecode_labels

# order of class labels returned in model.predict columns
classes_order = list(
    np.sort(np.array(list(prdtypecode_labels.keys()), dtype="str")).astype("int")
)

model_path_load = 'models/efficientnet_b3_001_phase2.keras'
model = None
initialized = False

def f1_score_sklearn(y_true, y_pred):
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    return f1_score(y_true_classes, y_pred_classes, average='weighted')

def w_f1_score(y_true, y_pred):
    return tf.py_function(f1_score_sklearn, (y_true, y_pred), tf.float64)

@st.cache_resource
def load_model(path):
    # Patch InputLayer to handle Keras 3 saved models (batch_shape → batch_input_shape)
    original_from_config = tf.keras.layers.InputLayer.from_config
    @classmethod
    def patched_from_config(cls, config):
        if 'batch_shape' in config:
            config['batch_input_shape'] = config.pop('batch_shape')
        return original_from_config.__func__(cls, config)
    tf.keras.layers.InputLayer.from_config = patched_from_config

    return tf.keras.models.load_model(
        path,
        custom_objects={'w_f1_score': w_f1_score},
        compile=False
    )

def init():
    global model, initialized
    model = load_model(model_path_load)
    initialized = True
    print("✅ Model loaded successfully")

def predict(image, prob_class_order):
    if not initialized:
        init()
    x = preprocess_input(np.expand_dims(image, axis=0))
    pred = model.predict(x)[0]
    reordered_pred = reorder_predict_cols(pred, classes_order, prob_class_order)
    return reordered_pred

def reorder_predict_cols(pred, old_ordered_classes, new_ordered_classes):
    old_classes_to_index = {}
    for i in range(len(old_ordered_classes)):
        old_classes_to_index[old_ordered_classes[i]] = i
    reordered_indexes = [old_classes_to_index[c] for c in new_ordered_classes]
    return pred[reordered_indexes]
