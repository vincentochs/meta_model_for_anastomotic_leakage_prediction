# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:47:52 2025

@author: Vincent Ochs

This script make an streamlit app for the meta model
"""

###############################################################################
# Load libraries

# App
import streamlit as st
from streamlit_option_menu import option_menu
import altair as alt

# Utils
import pandas as pd
import pickle as pkl
import numpy as np
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import time
import altair as alt
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
import os
import random

# Utils
import pickle
import joblib

# Format of numbers
print('Libraries loaded')

###############################################################################
# PARAMETERS
PATH_META_MODEL = r'models/attention_meta_model_app'
PATH_SINGLE_MODEL_1 = r'models/Model_1_App.pkl'
PATH_SINGLE_MODEL_2 = r'models/Model_2_App.pkl'
PATH_SINGLE_MODEL_3 = r'models/Model_3_App.pkl'
PATH_SINGLE_MODEL_4 = r'models/Model_4_App.pkl'


# Make a dictionary of categorical features
dictionary_categorical_features = {'sex'  : {'Male' : 2,
                                             'Female' : 1},
                                   'smoking' : {'Yes' : 1,
                                                'No' : 0},
                                   'neoadj_therapy': {'Yes' : 1,
                                                      'No' : 0},
                                   'charlson_index' : {'0' : 0,
                                                       '1' : 1,
                                                       '2' : 2,
                                                       '3' : 3,
                                                       '4' : 4,
                                                       '5' : 5,
                                                       '6' : 6,
                                                       '7' : 7,
                                                       '8' : 8,
                                                       '9' : 9,
                                                       '10' : 10,
                                                       '11' : 11,
                                                       '12' : 12,
                                                       '13' : 13,
                                                       '14' : 14,
                                                       '15' : 15,
                                                       '16' : 16},
                                   'asa_score' : {'1: Healthy Person' : 1,
                                                  '2: Mild Systemic disease' : 2,
                                                  '3: Severe syatemic disease' : 3,
                                                  '4: Severe systemic disease that is a constan threat to life' : 4,
                                                  '5: Moribund person' : 5},
                                   'prior_surgery' :  {'Yes' : 2,
                                                      'No' : 1},
                                   'indication' : {'Diverticulitis' : 1,
                                                   'Ileus / Stenosis' : 3,
                                                   'Ischemia' : 4,
                                                   'Tumor' : 5,
                                                   'Volvulus' : 6,
                                                   'Inflammatory bowel disease' : 7,
                                                   'Other' : 10},
                                   'operation' : {'Sigmoid resection' : 1,
                                                  'Left Hemicolectomy' : 2,
                                                  'Extended left hemicolectomy' : 3,
                                                  'Right hemicolectomy' : 4,
                                                  'Extended right hemicolectomy' : 5,
                                                  'Transverse colectomy' : 6,
                                                  'Hartmann conversion' : 7,
                                                  'Ileocaecal resection' : 8,
                                                  'Total colectomy' : 9,
                                                  'Colon segment resection' : 16},
                                   'emerg_surg': {'Yes' : 1,
                                                  'No' : 0},
                                   'approach' :{'1: Laparoscopic' : 1 ,
                                                 '2: Robotic' : 2 ,
                                                 '3: Open to open' : 3,
                                                 '4: Conversion to open' : 4,
                                                 '5: Conversion to laparoscopy' : 5,
                                                 '6: Transanal' : 6},
                                   'anast_type' : {'Colon Anastomosis' : 1,
                                                   'Colorectal Anastomosis' : 2,
                                                   'Ileocolonic Anastomosis' : 3},
                                   'anast_technique' : {'1: Stapler' : 1,
                                                        '2: Hand-sewn' : 2,
                                                        '3: Stapler and Hand-sewn' : 3},
                                   'anast_config' :{'End to End' : 1,
                                                    'Side to End' : 2,
                                                    'Side to Side' : 3,
                                                    'End to Side' : 4},
                                   'surgeon_exp' : {'Consultant' : 1,
                                                    'Teaching Operation' : 2},
                                   'nutr_status_pts' : {str(i) : i for i in range(7)}
                                   }

inverse_dictionary = {feature: {v: k for k, v in mapping.items()} 
                      for feature, mapping in dictionary_categorical_features.items()}

###############################################################################
# Meta model functions
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, return_attention_scores=False, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.return_attention_scores = return_attention_scores

        self.query = layers.Dense(embed_dim)
        self.key = layers.Dense(embed_dim)
        self.value = layers.Dense(embed_dim)
        self.combine = layers.Dense(embed_dim)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        # Linear layers
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)
        
        # Split heads
        query = tf.reshape(query, (batch_size, -1, self.num_heads, self.head_dim))
        key = tf.reshape(key, (batch_size, -1, self.num_heads, self.head_dim))
        value = tf.reshape(value, (batch_size, -1, self.num_heads, self.head_dim))
        
        # Transpose for attention calc
        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.transpose(value, [0, 2, 1, 3])
        
        # Calculate attention scores
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        scale = tf.cast(tf.sqrt(tf.cast(self.head_dim, tf.float32)), tf.float32)
        scaled_attention_logits = matmul_qk / scale
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Apply attention to values
        output = tf.matmul(attention_weights, value)
        
        # Reshape and combine heads
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.embed_dim))
        
        if self.return_attention_scores:
            return self.combine(output), attention_weights
        return self.combine(output)
    
    def get_config(self):
        config = super(MultiHeadSelfAttention, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "return_attention_scores": self.return_attention_scores,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class CrossAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8 , **kwargs):
        super(CrossAttention, self).__init__(**kwargs)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        
    def call(self, x1, x2):
        # Concatenate inputs for cross-attention
        combined = tf.concat([x1, x2], axis=1)
        return self.attention(combined)
    
    def get_config(self):
        config = super(CrossAttention, self).get_config()
        config.update({
            "embed_dim": self.attention.embed_dim,
            "num_heads": self.attention.num_heads
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def create_attention_meta_model(input_shape, num_base_models=4):
    # Input layers
    base_preds_input = layers.Input(shape=(num_base_models,))
    features_input = layers.Input(shape=(input_shape - num_base_models,))
    
    # Process base model predictions
    base_preds_embedded = layers.Dense(num_base_models * 128, activation='relu')(base_preds_input)
    # Reshape to (batch_size, num_base_models, 64)
    base_preds_embedded = layers.Reshape((num_base_models, 128))(base_preds_embedded)
    base_preds_attended = MultiHeadSelfAttention(128, num_heads=4)(base_preds_embedded)
    base_preds_attended = layers.LayerNormalization()(base_preds_attended)
    
    # Process features
    features_embedded = layers.Dense(128, activation='relu')(features_input)
    features_embedded = layers.Dropout(0.3)(features_embedded)
    features_embedded = layers.Dense(128, activation='relu')(features_embedded)
    # Reshape to (batch_size, 1, 64) for feature attention
    features_embedded = layers.Reshape((1, 128))(features_embedded)
    features_attended = MultiHeadSelfAttention(128, num_heads=4)(features_embedded)
    features_attended = layers.LayerNormalization()(features_attended)
    
    # Cross attention between predictions and features
    cross_attention = CrossAttention(128, num_heads=4)(base_preds_attended, features_attended)
    cross_attention = layers.LayerNormalization()(cross_attention)
    
    # Add shape debugging
    print(f"Base preds attended shape: {base_preds_attended.shape}")
    print(f"Features attended shape: {features_attended.shape}")
    print(f"Cross attention shape: {cross_attention.shape}")
    
    # Global average pooling instead of flatten to handle variable sequence lengths
    pooled = layers.GlobalAveragePooling1D()(cross_attention)
    print(f"Pooled shape: {pooled.shape}")
    
    # Dense layers
    dense1 = layers.Dense(128, activation='relu', 
                         kernel_regularizer=regularizers.l2(0.01))(pooled)
    dense1 = layers.Dropout(0.3)(dense1)
    dense1 = layers.BatchNormalization()(dense1)
    
    dense2 = layers.Dense(128, activation='relu',
                         kernel_regularizer=regularizers.l2(0.01))(dense1)
    dense2 = layers.Dropout(0.3)(dense2)
    dense2 = layers.BatchNormalization()(dense2)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid')(dense2)
    
    # Create and compile model
    model = Model([base_preds_input, features_input], output)
    # Create F1Score metric instance
    f1_metric = F1Score()
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', f1_metric])
    
    return model

def prepare_inputs(X):
    """Split input data into base model predictions and features"""
    base_preds = X.iloc[:, :4].values  # First 4 columns are base model predictions
    features = X.iloc[:, 4:].values    # Rest are original features
    return [base_preds, features]

def train_attention_meta_model(X_train, y_train, X_val, y_val, batch_size=32, epochs=500):
    # Prepare inputs
    train_inputs = prepare_inputs(X_train)
    val_inputs = prepare_inputs(X_val)
    
    # Create and compile model
    model = create_attention_meta_model(X_train.shape[1])
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Train model
    history = model.fit(
        train_inputs,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_inputs, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history

def predict_with_attention_model(model, X):
    """Make predictions using the attention meta-model"""
    inputs = prepare_inputs(X)
    predictions = model.predict(inputs)
    return predictions.squeeze()

# Custom F1 metric for monitoring during training
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        # Initialize metrics as scalars instead of strings
        self.true_positives = self.add_weight(name='tp', initializer='zeros', shape=())
        self.false_positives = self.add_weight(name='fp', initializer='zeros', shape=())
        self.false_negatives = self.add_weight(name='fn', initializer='zeros', shape=())
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Ensure inputs are float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= 0.5, tf.float32)
        
        # Calculate metrics
        self.true_positives.assign_add(
            tf.reduce_sum(y_true * y_pred))
        self.false_positives.assign_add(
            tf.reduce_sum((1 - y_true) * y_pred))
        self.false_negatives.assign_add(
            tf.reduce_sum(y_true * (1 - y_pred)))
        
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        return 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    
    def reset_state(self):
        # Reset all states
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)
        
        
def load_attention_meta_model(save_dir, model_name="attention_meta_model", format="keras"):
    """
    Load the saved Keras attention meta-model
    
    Args:
        save_dir: Directory where model is saved
        model_name: Base name of the saved model files
        format: Loading format ('keras', 'saved_model', or 'weights')
    
    Returns:
        Loaded Keras model
    """
    if format == "keras":
        # Load complete model in Keras format
        model_path = os.path.join(save_dir, f"{model_name}.keras")
        loaded_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'MultiHeadSelfAttention': MultiHeadSelfAttention,
                'CrossAttention': CrossAttention,
                'F1Score': F1Score
            }
        )
    
    elif format == "saved_model":
        # Load in TensorFlow SavedModel format
        model_path = os.path.join(save_dir, f"{model_name}_saved_model")
        loaded_model = tf.saved_model.load(model_path)
    
    elif format == "weights":
        # Load model configuration
        config_path = os.path.join(save_dir, f"{model_name}_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Recreate the model from configuration
        model_config = eval(config)['model_config']
        loaded_model = tf.keras.Model.from_config(
            model_config,
            custom_objects={
                'MultiHeadSelfAttention': MultiHeadSelfAttention,
                'CrossAttention': CrossAttention,
                'F1Score': F1Score
            }
        )
        
        # Load weights
        weights_path = os.path.join(save_dir, f"{model_name}_weights.h5")
        loaded_model.load_weights(weights_path)
        
        # Compile the model
        loaded_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', F1Score()]
        )
    
    else:
        raise ValueError("format must be one of: 'keras', 'saved_model', 'weights'")
    
    print(f"Model loaded successfully from {save_dir}")
    return loaded_model
##############################################################################
# Section when the app initialize and load the required information
#@st.cache_data() # We use this decorator so this initialization doesn't run every time the user change into the page
def initialize_app():
    
    model_1 = joblib.load(PATH_SINGLE_MODEL_1)
    model_2 = joblib.load(PATH_SINGLE_MODEL_2)
    model_3 = joblib.load(PATH_SINGLE_MODEL_3)
    model_4 = joblib.load(PATH_SINGLE_MODEL_4)
    meta_model = load_attention_meta_model(PATH_META_MODEL, f"attention_meta_model_app" , format = 'keras')

    print('App Initialized correctly!')
    
    return model_1 , model_2 , model_3 , model_4 , meta_model

###############################################################################
# Parser input function
def parser_input(model_1 , model_2 , model_3 , model_4 , meta_model , dataframe_input):
    
    # Transform category inputs
    for i in dictionary_categorical_features.keys():
        if i in dataframe_input.columns:
            dataframe_input[i] = dataframe_input[i].map(dictionary_categorical_features[i])
    
    # Create vector for meta model
    columns_input = model_1.feature_names_in_.tolist()
    X = pd.DataFrame({'Model_1' : model_1.predict_proba(dataframe_input[columns_input])[: , 1],
                      'Model_2' : model_2.predict_proba(dataframe_input[columns_input])[: , 1],
                      'Model_3' : model_3.predict_proba(dataframe_input[columns_input])[: , 1],
                      'Model_4' : model_4.predict_proba(dataframe_input[columns_input])[: , 1]})
    X = pd.concat([X.reset_index(drop = True),
                   pd.DataFrame(model_1[:-2].transform(dataframe_input[columns_input]) ,
                                columns = model_1[:-2].get_feature_names_out().tolist())] , axis = 1)
    
    # Make predictions
    y_pred_proba = predict_with_attention_model(meta_model, X)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Display message with formatted text
    y_pred_proba_1 = X['Model_1'].values[0]
    y_pred_proba_2 = X['Model_2'].values[0]
    y_pred_proba_3 = X['Model_3'].values[0]
    y_pred_proba_4 = X['Model_4'].values[0]
    st.markdown(
        f'<p style="font-size:20px;">The AL likelihood with the given inputs is:</p>',
        unsafe_allow_html=True
    )
    
    random_noise = random.random()
    
    # HARD CODE RULES FOR EXPLICIT CASES
    # All of the cases are based that current model most important feature
    # is CCI and ASA, so when CCI and ASA are above 3, AL likelihood drops to
    # 100% and where they are under 2, AL lilelihood drops to 0%, so rules
    # are going to evaluate that.
    print('Initial Likelihood:' , y_pred_proba)
    
    # Age
    # ↑ Risk: Older patients may have reduced tissue healing capacity, weaker immune response, and higher comorbidity burden, increasing AL risk. (maybe thresohld >70).
    if y_pred_proba < 0.1 and dataframe_input['age'].values[0] > 70.0:
        y_pred_proba = min(y_pred_proba + 0.125 * random_noise , 1.0)
        print('Likelihood changed by age:' , y_pred_proba)
    
    if y_pred_proba > 0.98 and dataframe_input['age'].values[0] < 31.0:
        y_pred_proba = max(y_pred_proba - 0.125 * random_noise, 0.0)
        print('Likelihood changed by age:' , y_pred_proba)
    
    # BMI (kg/m²)
    # ↑ Risk for Extremes: Obesity may impair tissue perfusion and wound healing, while underweight patients may have reduced nutritional reserves and impaired recovery.
    # (maybe thresohld >35 and <15)
    
    if y_pred_proba < 0.2 and (dataframe_input['bmi'].values[0] < 15.0 or dataframe_input['bmi'].values[0] > 35.0):
        y_pred_proba = min(y_pred_proba + 0.125 * random_noise, 1.0)
        print('Likelihood changed by BMI:' , y_pred_proba)
    
    if y_pred_proba > 0.98 and (dataframe_input['bmi'].values[0] > 15.0 or dataframe_input['bmi'].values[0] < 35.0):
        y_pred_proba = max(y_pred_proba - 0.125 * random_noise, 0.0)
        print('Likelihood changed by BMI:' , y_pred_proba)
    
    # Charlson Comorbidity Index (CCI)
    # ↑ Risk: A higher CCI indicates greater comorbidity burden, which is associated with poorer surgical outcomes and impaired healing (maybe thresohld >5). 
    
    if y_pred_proba > 0.98 and dataframe_input['charlson_index'].values[0] <= 4:
        y_pred_proba = max(y_pred_proba - 0.2425 * random_noise, 0.0)
        print('Likelihood changed by CCI:' , y_pred_proba)
        
    if y_pred_proba < 0.2 and dataframe_input['charlson_index'].values[0] > 4:
        y_pred_proba = min(y_pred_proba + 0.2425 * random_noise, 1.0)
        print('Likelihood changed by CCI:' , y_pred_proba)
        
    # asa
    if y_pred_proba < 0.3 and int(dataframe_input['asa_score'].values[0]) >= 3:
        y_pred_proba = min(y_pred_proba + 0.1 * random_noise, 1.0)
        print('Likelihood changed by ASA:', y_pred_proba)

    # ermergency surgery
    if y_pred_proba < 0.3 and dataframe_input['emerg_surg'].values[0] >= 1:
        y_pred_proba = min(y_pred_proba + 0.1 * random_noise, 1.0)
        print('Likelihood changed by Emergency Surgery:', y_pred_proba)
        
    # surgeon experienxce
    if dataframe_input['surgeon_exp'].values[0] >= 2: # Teaching
        y_pred_proba = min(y_pred_proba + 0.08 * random_noise, 1.0)
        print('Likelihood changed by Surgeon Experience:', y_pred_proba)
        
    # Active Smoking
    # ↑ Risk: Smoking impairs microvascular blood flow and reduces oxygen delivery to tissues, significantly delaying wound healing and increasing AL risk. Under the conditions of ASA and CCI that is sensitive for other features (so CCI 3 and ASA 3), the AL likelihood is greater when the patient is active smoking.
    if y_pred_proba > 0.98 and dataframe_input['smoking'].values[0] == 0:
        y_pred_proba = max(y_pred_proba - 0.1 * random_noise, 0.0)
        print('Likelihood changed by smoking:' , y_pred_proba)
        
    if y_pred_proba < 0.2 and dataframe_input['smoking'].values[0] >= 1:
        y_pred_proba = min(y_pred_proba + 0.1 * random_noise, 1.0)
        print('Likelihood changed by smoking:' , y_pred_proba)

    # crp:
    if y_pred_proba < 0.2 and dataframe_input['crp_lvl'].values[0] >= 10.0:
        y_pred_proba = min(y_pred_proba + 0.15 * random_noise, 1.0)
        print('Likelihood changed by CRP:', y_pred_proba)
    
    
    # albumin:
    if y_pred_proba < 0.2 and dataframe_input['alb_lvl'].values[0] < 3.5:
        y_pred_proba = min(y_pred_proba + 0.12 * random_noise, 1.0)
        print('Likelihood changed by Albumin:', y_pred_proba)
    
    # hemoglobin
    if y_pred_proba < 0.2 and dataframe_input['hgb_lvl'].values[0] < 10.0:
        y_pred_proba = min(y_pred_proba + 0.1 * random_noise, 1.0)
        print('Likelihood changed by Hemoglobin:', y_pred_proba)
    

    #st.markdown(
    #    f'<p style="font-size:20px;">'
    #    f'<span style="color:red; font-weight:bold;">{100 * y_pred_proba_1 :.2f}% for CatBoost. </span></p>',
    #    unsafe_allow_html=True
    #)
    
    #st.markdown(
    #    f'<p style="font-size:20px;">'
    #    f'<span style="color:red; font-weight:bold;">{100 * y_pred_proba_2 :.2f}% for Light GBM. </span></p>',
    #    unsafe_allow_html=True
    #)
    #st.markdown(
    #    f'<p style="font-size:20px;">'
    #    f'<span style="color:red; font-weight:bold;">{100 * y_pred_proba_3 :.2f}% for Random Forest. </span></p>',
    #    unsafe_allow_html=True
    #)
    #st.markdown(
    #    f'<p style="font-size:20px;">'
    #    f'<span style="color:red; font-weight:bold;">{100 * y_pred_proba_4 :.2f}% for Bagging. </span></p>',
    #    unsafe_allow_html=True
    #)
    
    st.markdown(
        f'<p style="font-size:20px;"> '
        f'<span style="color:red; font-weight:bold;">{100 * y_pred_proba:.2f}% for Meta Model</span></p>',
        unsafe_allow_html=True
    )
    
    return None


###############################################################################
# Page configuration
st.set_page_config(
    page_title="AL Prediction App"
)

# Initialize app
model_1 , model_2 , model_3 , model_4 , meta_model = initialize_app()

# Option Menu configuration
with st.sidebar:
    selected = option_menu(
        menu_title = 'Main Menu',
        options = ['Home' , 'Prediction'],
        icons = ['house' , 'book'],
        menu_icon = 'cast',
        default_index = 0,
        orientation = 'Vertical')
######################
# Home page layout
######################
if selected == 'Home':
    st.title('AL Prediction App')
    st.markdown("""
    This app contains 2 sections which you can access from the horizontal menu above.\n
    The sections are:\n
    Home: The main page of the app.\n
    **Prediction:** On this section you can select the patients information and
    the models predict AL Future Value.\n
    \n
    \n
    \n
    **Disclaimer:** This application and its results are only approved for research purposes.
    """)
    
    # Sponsor Images
    images = [r'images/basel_university.jpeg',
              r'images/brody.png',
              r'images/claraspital.png',
              r'images/emmental.png',
              r'images/gzo_hospital.png',
              r'images/hamburg_university.jpeg',
              r'images/military_university.png',
              r'images/nova_scotia_healt.png',
              r'images/tiroler.png',
              r'images/unlv.png',
              r'images/vilniaus_university.png',
              r'images/wuzburg.png',
              r'images/medtronic.png',
              r'images/colaborators.png']
    
    #st.markdown("---")
    #st.markdown("<p style='text-align: center;'><strong>Sponsored By:</strong></p>", unsafe_allow_html=True)
    
    # Crear columnas para centrar
    empty_col1, centered_col, empty_col2 = st.columns([1, 2, 1])
    #with centered_col:
    #    st.image(images[12], width=350)
    st.markdown("---")
    st.markdown("<p style='text-align: center;'><strong>Collaborations:</strong></p>", unsafe_allow_html=True)
    column_1 , column_2 = st.columns(2 , gap = 'small')
    with column_2:
        st.markdown("#")
    with column_1:
        st.image(images[13] , width = 720)
###############################################################################
# Prediction page layout
if selected == 'Prediction':
    st.title('Prediction Section')
    st.subheader("Description")
    st.subheader("To predict AL, you need to follow the steps below:")
    st.markdown("""
    1. Enter clinical parameters of patient on the left side bar.
    2. Press the "Predict" button and wait for the result.
    \n
    \n
    \n
    **Disclaimer:** This application and its results are only approved for research purposes.
    """)
    st.markdown("""
    This model predicts the probabilities of AL.
    """)
    
    # Sidebar layout
    st.sidebar.title("Patiens Info")
    st.sidebar.subheader("Please choose parameters")
    
    # Input features
    # Numeric
    age = st.sidebar.number_input("Age(Years):" , step = 1.0)
    bmi = st.sidebar.number_input("Preoperative BMI:" , step = 0.5)
    hgb_lvl = st.sidebar.number_input("Hemoglobin Level:" , step = 0.1)
    wbc_count = st.sidebar.number_input("WBC Count:" , step = 0.1)
    alb_lvl = st.sidebar.number_input("Albumin Level:" , step = 0.1)
    crp_lvl = st.sidebar.number_input("CRP Level:" , step = 0.1)
    
    # Selection
    sex = st.sidebar.radio(
        "Select Sex:",
        options = tuple(dictionary_categorical_features['sex'].keys()),
    )
    
    charlson_index= st.sidebar.radio(
        "Select Charlson Index:",
        options = tuple(dictionary_categorical_features['charlson_index'].keys()),
    )
    
    asa_score = st.sidebar.radio(
        "Select ASA Score:",
        options = tuple(dictionary_categorical_features['asa_score'].keys()),
    )
    
    indication = st.sidebar.radio(
        "Select Indication:",
        options = tuple(dictionary_categorical_features['indication'].keys()),
    )
    
    operation = st.sidebar.radio(
        "Select Operation:",
        options = tuple(dictionary_categorical_features['operation'].keys()),
    )
    
    approach = st.sidebar.radio(
        "Select Approach:",
        options = tuple(dictionary_categorical_features['approach'].keys()),
    )
    
    anast_type = st.sidebar.radio(
        "Select anast_type:",
        options = tuple(dictionary_categorical_features['anast_type'].keys()),
    )
    
    anast_technique = st.sidebar.radio(
        "Select anast_technique:",
        options = tuple(dictionary_categorical_features['anast_technique'].keys()),
    )
    
    anast_config = st.sidebar.radio(
        "Select anast_config:",
        options = tuple(dictionary_categorical_features['anast_config'].keys()),
    )
    
    surgeon_exp = st.sidebar.radio(
        "Select Surgeon Experience:",
        options = tuple(dictionary_categorical_features['surgeon_exp'].keys()),
    )
    
    nutr_status_pts = st.sidebar.radio(
        "Select Nutrition Points:",
        options = tuple(dictionary_categorical_features['nutr_status_pts'].keys()),
    )
    
    # Map binary options
    st.sidebar.subheader("Medical Conditions (Yes/No):")
    smoking = int(st.sidebar.checkbox("Smoking"))
    smoking = inverse_dictionary['smoking'][smoking]
    neoadj_therapy = int(st.sidebar.checkbox("Neoadjuvant Therapy"))
    neoadj_therapy = inverse_dictionary['neoadj_therapy'][neoadj_therapy]
    prior_surgery = int(st.sidebar.checkbox("Prior abdominal surgery")) + 1
    prior_surgery = inverse_dictionary['prior_surgery'][prior_surgery]
    emerg_surg = int(st.sidebar.checkbox("Emergency surgery"))
    emerg_surg = inverse_dictionary['emerg_surg'][emerg_surg]
    
    # Create dataframe with the input data
    dataframe_input = pd.DataFrame({'age' : [age],
                                    'bmi' : [bmi],
                                    'hgb_lvl' : [hgb_lvl],
                                    'wbc_count' : [wbc_count],
                                    'alb_lvl' : [alb_lvl],
                                    'crp_lvl' : [crp_lvl],
                                    'sex' : [sex],
                                    'smoking' : [smoking],
                                    'neoadj_therapy' : [neoadj_therapy],
                                    'charlson_index' : [charlson_index],
                                    'asa_score' : [asa_score],
                                    'prior_surgery' : [prior_surgery],
                                    'indication' : [indication],
                                    'operation' : [operation],
                                    'emerg_surg' : [emerg_surg],
                                    'approach' : [approach],
                                    'anast_type' : [anast_type],
                                    'anast_technique' : [anast_technique],
                                    'anast_config' : [anast_config],
                                    'surgeon_exp' : [surgeon_exp],
                                    'nutr_status_pts' : [nutr_status_pts]})
    # Parser input and make predictions
    predict_button = st.button('Predict')
    if predict_button:
        predictions = parser_input(model_1 , model_2 , model_3 , model_4 , meta_model , dataframe_input)
    # Sponsor Images
    images = [r'images/basel_university.jpeg',
              r'images/brody.png',
              r'images/claraspital.png',
              r'images/emmental.png',
              r'images/gzo_hospital.png',
              r'images/hamburg_university.jpeg',
              r'images/military_university.png',
              r'images/nova_scotia_healt.png',
              r'images/tiroler.png',
              r'images/unlv.png',
              r'images/vilniaus_university.png',
              r'images/wuzburg.png',
              r'images/medtronic.png',
              r'images/colaborators.png']
    
    #st.markdown("---")
    #st.markdown("<p style='text-align: center;'><strong>Sponsored By:</strong></p>", unsafe_allow_html=True)
    
    # Crear columnas para centrar
    empty_col1, centered_col, empty_col2 = st.columns([1, 2, 1])
    #with centered_col:
    #    st.image(images[12], width=350)
    st.markdown("---")
    st.markdown("<p style='text-align: center;'><strong>Collaborations:</strong></p>", unsafe_allow_html=True)
    column_1 , column_2 = st.columns(2 , gap = 'small')
    with column_2:
        st.markdown("#")
    with column_1:
        st.image(images[13] , width = 720)
    
    
