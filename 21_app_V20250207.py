# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:47:52 2025

@author: Vincent Ochs

This script makes an streamlit app for the meta model
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
random_noise = 0.5

# Risk thresholds
RISK_THRESHOLDS = {
    'age_high': 70,
    'age_low': 30,
    'bmi_high': 35,
    'bmi_low': 18.5,
    'albumin_low': 3.5,
    'cci_high': 3,
    'asa_high': 3,
    'crp_high': 10.0,
    'hgb_low': 10.0
}

# Risk multipliers (how much to adjust probability)
RISK_MULTIPLIERS = {
    'age': 0.125,
    'bmi': 0.125,
    'albumin': 0.12,
    'cci': 0.1,
    'asa': 0.1,
    'smoking': 0.1,
    'neoadj_therapy': 0.08,
    'prior_surgery': 0.09,
    'emergency_surgery': 0.1,
    'surgeon_exp': 0.08,
    'approach_open': 0.07,
    'crp': 0.15,
    'hemoglobin': 0.1
}

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
                                                       '16' : 16,
                                                       'Unknown' : -1},
                                   'asa_score' : {'1: Healthy Person' : 1,
                                                  '2: Mild Systemic disease' : 2,
                                                  '3: Severe Systemic disease' : 3,
                                                  '4: Severe Systemic disease that is a constant threat to life' : 4,
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
                                                 '3: Open' : 3,
                                                 '4: Conversion to open' : 4,
                                                 '5: Conversion to laparoscopy' : 5,
                                                 '6: Transanal' : 6},
                                   'anast_type' : {'Colon Anastomosis' : 1,
                                                   #'Colorectal Anastomosis' : 2,
                                                   'Ileocolonic Anastomosis' : 3},
                                   'anast_technique' : {'1: Stapler' : 1,
                                                        '2: Hand-sewn' : 2},
                                   'anast_config' :{'End to End' : 1,
                                                    'Side to End' : 2,
                                                    'Side to Side' : 3,
                                                    'End to Side' : 4,
                                                    'Unknown' : -1},
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
@st.cache_resource
def load_single_models():
    model_1 = joblib.load(PATH_SINGLE_MODEL_1)
    model_2 = joblib.load(PATH_SINGLE_MODEL_2)
    model_3 = joblib.load(PATH_SINGLE_MODEL_3)
    model_4 = joblib.load(PATH_SINGLE_MODEL_4)
    return model_1, model_2, model_3, model_4

@st.cache_resource
def load_meta_model():
    return load_attention_meta_model(PATH_META_MODEL, "attention_meta_model_app", format='keras')

@st.cache_resource
def initialize_app():
    model_1, model_2, model_3, model_4 = load_single_models()
    meta_model = load_meta_model()
    print('App Initialized correctly!')
    return model_1, model_2, model_3, model_4, meta_model

###############################################################################
# Parser input function
def parser_input(model_1 , model_2 , model_3 , model_4 , meta_model , dataframe_input):
    
    # Handle mapping for categorical features, preserving -1 for "Not Available"
    for col_name in dictionary_categorical_features.keys():
        if col_name in dataframe_input.columns:
            value = dataframe_input.at[0, col_name]
            # If value is a string (e.g., 'Male'), map it. If it's -1, leave it.
            if isinstance(value, str) and value in dictionary_categorical_features[col_name]:
                dataframe_input.at[0, col_name] = dictionary_categorical_features[col_name][value]
    
    # Create vector for meta model
    columns_input = model_1.feature_names_in_.tolist()
    
    # Fill -1 values with a strategy (e.g., mean/median) before predicting if models can't handle it
    # For now, we assume the preprocessing pipeline inside the model handles it.
    df_for_prediction = dataframe_input.copy()

    X = pd.DataFrame({'Model_1' : model_1.predict_proba(df_for_prediction[columns_input])[: , 1],
                      'Model_2' : model_2.predict_proba(df_for_prediction[columns_input])[: , 1],
                      'Model_3' : model_3.predict_proba(df_for_prediction[columns_input])[: , 1],
                      'Model_4' : model_4.predict_proba(df_for_prediction[columns_input])[: , 1]})
    X = pd.concat([X.reset_index(drop = True),
                   pd.DataFrame(model_1[:-2].transform(df_for_prediction[columns_input]) ,
                                columns = model_1[:-2].get_feature_names_out().tolist())] , axis = 1)
    
    # Make predictions
    y_pred_proba = predict_with_attention_model(meta_model, X)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    st.markdown(
        f'<p style="font-size:20px;">The AL likelihood with the given inputs is:</p>',
        unsafe_allow_html=True
    )
    
    # PARAMETRIC RISK ADJUSTMENT RULES
    print('Initial Likelihood:', y_pred_proba)

    # Added checks for -1 (Not Available) before applying any rule.
    
    # Age risk adjustment
    age_val = dataframe_input['age'].values[0]
    if age_val != -1:
        if age_val > RISK_THRESHOLDS['age_high']:
            y_pred_proba = min(y_pred_proba + RISK_MULTIPLIERS['age'] * random_noise, 1.0)
            print('Likelihood increased by age:', y_pred_proba)
        elif y_pred_proba > 0.98 and age_val < RISK_THRESHOLDS['age_low']:
            y_pred_proba = max(y_pred_proba - RISK_MULTIPLIERS['age'] * random_noise, 0.0)
            print('Likelihood decreased by young age:', y_pred_proba)
    
    # BMI risk adjustment
    bmi_val = dataframe_input['bmi'].values[0]
    if bmi_val != -1 and (bmi_val < RISK_THRESHOLDS['bmi_low'] or bmi_val > RISK_THRESHOLDS['bmi_high']):
        y_pred_proba = min(y_pred_proba + RISK_MULTIPLIERS['bmi'] * random_noise, 1.0)
        print('Likelihood increased by BMI extremes:', y_pred_proba)
    
    # Albumin risk adjustment
    alb_val = dataframe_input['alb_lvl'].values[0]
    if alb_val != -1 and alb_val < RISK_THRESHOLDS['albumin_low']:
        y_pred_proba = min(y_pred_proba + RISK_MULTIPLIERS['albumin'] * random_noise, 1.0)
        print('Likelihood increased by low albumin:', y_pred_proba)
    
    # Charlson Comorbidity Index
    cci_val = int(dataframe_input['charlson_index'].values[0])
    if cci_val != -1:
        if cci_val > RISK_THRESHOLDS['cci_high']:
            y_pred_proba = min(y_pred_proba + RISK_MULTIPLIERS['cci'] * random_noise * cci_val, 1.0)
            print('Likelihood increased by high CCI:', y_pred_proba)
        elif y_pred_proba > 0.98 and cci_val <= RISK_THRESHOLDS['cci_high']:
            y_pred_proba = max(y_pred_proba - RISK_MULTIPLIERS['cci'] * random_noise, 0.0)
            print('Likelihood decreased by low CCI:', y_pred_proba)
    
    # ASA Score
    asa_val = int(dataframe_input['asa_score'].values[0])
    if asa_val != -1 and asa_val >= RISK_THRESHOLDS['asa_high']:
        y_pred_proba = min(y_pred_proba + RISK_MULTIPLIERS['asa'] * random_noise * asa_val, 1.0)
        print('Likelihood increased by high ASA:', y_pred_proba)
    
    # Smoking
    smoking_val = dataframe_input['smoking'].values[0]
    if smoking_val != -1:
        if smoking_val >= 1: # Yes
            y_pred_proba = min(y_pred_proba + RISK_MULTIPLIERS['smoking'] * random_noise, 1.0)
            print('Likelihood increased by smoking:', y_pred_proba)
        elif y_pred_proba > 0.98 and smoking_val == 0: # No
            y_pred_proba = max(y_pred_proba - RISK_MULTIPLIERS['smoking'] * random_noise, 0.0)
            print('Likelihood decreased by non-smoking:', y_pred_proba)
    
    # Neoadjuvant therapy
    neoadj_val = dataframe_input['neoadj_therapy'].values[0]
    if neoadj_val != -1 and neoadj_val >= 1:
        y_pred_proba = min(y_pred_proba + RISK_MULTIPLIERS['neoadj_therapy'] * random_noise, 1.0)
        print('Likelihood increased by neoadjuvant therapy:', y_pred_proba)
    
    # Prior abdominal surgery
    prior_surg_val = dataframe_input['prior_surgery'].values[0]
    if prior_surg_val != -1 and prior_surg_val >= 2:  # 'Yes' is mapped to 2
        y_pred_proba = min(y_pred_proba + RISK_MULTIPLIERS['prior_surgery'] * random_noise, 1.0)
        print('Likelihood increased by prior surgery:', y_pred_proba)
    
    # Emergency surgery
    emerg_surg_val = dataframe_input['emerg_surg'].values[0]
    if emerg_surg_val != -1 and emerg_surg_val >= 1:
        y_pred_proba = min(y_pred_proba + RISK_MULTIPLIERS['emergency_surgery'] * random_noise, 1.0)
        print('Likelihood increased by emergency surgery:', y_pred_proba)
    
    # Surgical approach
    approach_val = int(dataframe_input['approach'].values[0])
    if approach_val != -1 and approach_val in [3, 4]:  # Open or Conversion to open
        y_pred_proba = min(y_pred_proba + RISK_MULTIPLIERS['approach_open'] * random_noise, 1.0)
        print('Likelihood increased by open approach:', y_pred_proba)
    
    # Surgeon experience
    surgeon_exp_val = dataframe_input['surgeon_exp'].values[0]
    if surgeon_exp_val != -1 and surgeon_exp_val >= 2:  # Teaching operation
        y_pred_proba = min(y_pred_proba + RISK_MULTIPLIERS['surgeon_exp'] * random_noise, 1.0)
        print('Likelihood increased by teaching operation:', y_pred_proba)
    
    # CRP level
    crp_val = dataframe_input['crp_lvl'].values[0]
    if crp_val != -1 and crp_val >= RISK_THRESHOLDS['crp_high']:
        y_pred_proba = min(y_pred_proba + RISK_MULTIPLIERS['crp'] * random_noise, 1.0)
        print('Likelihood increased by high CRP:', y_pred_proba)
    
    # Hemoglobin level
    hgb_val = dataframe_input['hgb_lvl'].values[0]
    if hgb_val != -1 and hgb_val < RISK_THRESHOLDS['hgb_low']:
        y_pred_proba = min(y_pred_proba + RISK_MULTIPLIERS['hemoglobin'] * random_noise, 1.0)
        print('Likelihood increased by low hemoglobin:', y_pred_proba)
          
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
    This app contains 2 sections which you can access from the sidebar menu on the left..\n
    The sections are:\n
    Home: The main page of the app.\n
    **Prediction:** On this section you can select the patients information and
    run the models to predict AL risk.\n
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
    1. Enter the patient’s clinical parameters on the left side bar. If a parameter is unknown, check the corresponding "Not Available" box.
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
    st.sidebar.title("Patient Info")
    st.sidebar.subheader("Please choose parameters")
    
    # MODIFICATION: Added a "Not Available" checkbox for each input.
    # The value will be set to -1 if the box is checked.
    
    # Numeric Inputs
    age_na = st.sidebar.checkbox("Age: Not Available")
    if not age_na:
        age = st.sidebar.number_input("Age (Years):", step=1.0, disabled=age_na, value=40.0)
    if age_na: age = -1

    bmi_na = st.sidebar.checkbox("Preoperative BMI: Not Available")
    if not bmi_na:
        bmi = st.sidebar.number_input("Preoperative BMI:", step=0.5, disabled=bmi_na, value=25.0)
    if bmi_na: bmi = -1
    
    hgb_lvl_na = st.sidebar.checkbox("Hemoglobin Level: Not Available")
    if not hgb_lvl_na:
        hgb_lvl = st.sidebar.number_input("Hemoglobin Level (g/dL):", step=0.1, disabled=hgb_lvl_na, value=12.0)
    if hgb_lvl_na: hgb_lvl = -1

    wbc_count_na = st.sidebar.checkbox("White blood cell count (WBC): Not Available")
    if not wbc_count_na:
        wbc_count = st.sidebar.number_input("White blood cell count (WBC) (10³/µL):", step=0.1, disabled=wbc_count_na, value=7.0)
    if wbc_count_na: wbc_count = -1
    
    alb_lvl_na = st.sidebar.checkbox("Albumin Level: Not Available")
    if not alb_lvl_na:
        alb_lvl = st.sidebar.number_input("Albumin Level (g/dL):", step=0.1, disabled=alb_lvl_na, value=4.0)
    if alb_lvl_na: alb_lvl = -1

    crp_lvl_na = st.sidebar.checkbox("CRP Level: Not Available")
    if not crp_lvl_na:
        crp_lvl = st.sidebar.number_input("CRP Level (mg/L):", step=0.1, disabled=crp_lvl_na, value=5.0)
    if crp_lvl_na: crp_lvl = -1

    # Selection Inputs
    st.sidebar.markdown("---")
    sex_na = st.sidebar.checkbox("Sex: Not Available")
    if not sex_na:
        sex = st.sidebar.radio("Select Sex:", options=tuple(dictionary_categorical_features['sex'].keys()))
    if sex_na: sex = -1
    
    # NOTE: Charlson Index already has 'Unknown' which maps to -1, so no NA box is needed.
    charlson_index = st.sidebar.radio("Select Charlson Comorbidity Index (CCI):", options=tuple(dictionary_categorical_features['charlson_index'].keys()))
    
    asa_score_na = st.sidebar.checkbox("ASA Score: Not Available")
    if not asa_score_na:
        asa_score = st.sidebar.radio("Select ASA Score:", options=tuple(dictionary_categorical_features['asa_score'].keys()))
    if asa_score_na: asa_score = -1

    indication_na = st.sidebar.checkbox("Indication: Not Available")
    if not indication_na:
        indication = st.sidebar.radio("Select Indication:", options=tuple(dictionary_categorical_features['indication'].keys()))
    if indication_na: indication = -1

    operation_na = st.sidebar.checkbox("Operation: Not Available")
    if not operation_na:
        operation = st.sidebar.radio("Select Operation:", options=tuple(dictionary_categorical_features['operation'].keys()))
    if operation_na: operation = -1

    approach_na = st.sidebar.checkbox("Approach: Not Available")
    if not approach_na:
        approach = st.sidebar.radio("Select Approach:", options=tuple(dictionary_categorical_features['approach'].keys()))
    if approach_na: approach = -1

    anast_type_na = st.sidebar.checkbox("Anastomotic Type: Not Available")
    if not anast_type_na:
        anast_type = st.sidebar.radio("Select Anastomotic Type:", options=tuple(dictionary_categorical_features['anast_type'].keys()))
    if anast_type_na: anast_type = -1

    anast_technique_na = st.sidebar.checkbox("Anastomotic Technique: Not Available")
    if not anast_technique_na:
        anast_technique = st.sidebar.radio("Select Anastomotic Technique:", options=tuple(dictionary_categorical_features['anast_technique'].keys()))
    if anast_technique_na: anast_technique = -1

    # NOTE: Anastomotic Configuration already has 'Unknown' which maps to -1.
    anast_config = st.sidebar.radio("Select Anastomotic Configuration:", options=tuple(dictionary_categorical_features['anast_config'].keys()))
    
    surgeon_exp_na = st.sidebar.checkbox("Surgeon Experience: Not Available")
    if not surgeon_exp_na:
        surgeon_exp = st.sidebar.radio("Select Surgeon Experience:", options=tuple(dictionary_categorical_features['surgeon_exp'].keys()))
    if surgeon_exp_na: surgeon_exp = -1

    nutr_status_pts_na = st.sidebar.checkbox("Nutritional Risk Screening (NRS): Not Available")
    if not nutr_status_pts_na:
        nutr_status_pts = st.sidebar.radio("Select Nutritional Risk Screening (NRS):", options=tuple(dictionary_categorical_features['nutr_status_pts'].keys()))
    if nutr_status_pts_na: nutr_status_pts = -1

    # Binary options
    st.sidebar.markdown("---")
    st.sidebar.subheader("Medical Conditions (Yes/No):")

    smoking_na = st.sidebar.checkbox("Smoking: Not Available")
    smoking_yes = st.sidebar.checkbox("Smoking", disabled=smoking_na)
    smoking = -1 if smoking_na else ("Yes" if smoking_yes else "No")

    neoadj_therapy_na = st.sidebar.checkbox("Neoadjuvant Therapy: Not Available")
    neoadj_therapy_yes = st.sidebar.checkbox("Neoadjuvant Therapy", disabled=neoadj_therapy_na)
    neoadj_therapy = -1 if neoadj_therapy_na else ("Yes" if neoadj_therapy_yes else "No")
    
    prior_surgery_na = st.sidebar.checkbox("Prior abdominal surgery: Not Available")
    prior_surgery_yes = st.sidebar.checkbox("Prior abdominal surgery", disabled=prior_surgery_na)
    prior_surgery = -1 if prior_surgery_na else ("Yes" if prior_surgery_yes else "No")

    emerg_surg_na = st.sidebar.checkbox("Emergency surgery: Not Available")
    emerg_surg_yes = st.sidebar.checkbox("Emergency surgery", disabled=emerg_surg_na)
    emerg_surg = -1 if emerg_surg_na else ("Yes" if emerg_surg_yes else "No")
    
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
    
    predict_button = st.button('Predict')
    if predict_button:
        predictions = parser_input(model_1 , model_2 , model_3 , model_4 , meta_model , dataframe_input)
    
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
    
    st.markdown("---")
    st.markdown("<p style='text-align: center;'><strong>Collaborations:</strong></p>", unsafe_allow_html=True)
    column_1 , column_2 = st.columns(2 , gap = 'small')
    with column_2:
        st.markdown("#")
    with column_1:
        st.image(images[13] , width = 720)



