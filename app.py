import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import base64

# Function to convert image to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Load and preprocess data
data = pd.read_csv('https://raw.githubusercontent.com/adil200/Medical-Diagnoser/main/medical_data.csv')

# Tokenize the text data
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['Patient_Problem'])
sequences = tokenizer.texts_to_sequences(data['Patient_Problem'])
max_length = max(len(x) for x in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Encode the labels
label_encoder_disease = LabelEncoder()
label_encoder_prescription = LabelEncoder()
disease_labels = label_encoder_disease.fit_transform(data['Disease'])
prescription_labels = label_encoder_prescription.fit_transform(data['Prescription'])
disease_labels_categorical = to_categorical(disease_labels)
prescription_labels_categorical = to_categorical(prescription_labels)
Y = np.hstack((disease_labels_categorical, prescription_labels_categorical))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, Y, test_size=0.2, random_state=42)
y_train_disease = y_train[:, :len(label_encoder_disease.classes_)]
y_train_prescription = y_train[:, len(label_encoder_disease.classes_):]
y_test_disease = y_test[:, :len(label_encoder_disease.classes_)]
y_test_prescription = y_test[:, len(label_encoder_disease.classes_):]

# Define the model
input_layer = Input(shape=(max_length,))
embedding = Embedding(input_dim=5000, output_dim=64)(input_layer)
lstm_layer = LSTM(64)(embedding)
dropout_layer = Dropout(0.5)(lstm_layer)  # Add Dropout

disease_output = Dense(len(label_encoder_disease.classes_), activation='softmax', name='disease_output')(dropout_layer)
prescription_output = Dense(len(label_encoder_prescription.classes_), activation='softmax', name='prescription_output')(dropout_layer)

model = Model(inputs=input_layer, outputs=[disease_output, prescription_output])
model.compile(
    loss={'disease_output': 'categorical_crossentropy', 'prescription_output': 'categorical_crossentropy'},
    optimizer='adam',
    metrics={'disease_output': ['accuracy'], 'prescription_output': ['accuracy']}
)

# Train the model (Uncomment this line if training is needed)
# model.fit(X_train, {'disease_output': y_train_disease, 'prescription_output': y_train_prescription}, epochs=100, batch_size=32)

# Define prediction function
def make_prediction(patient_problem):
    sequence = tokenizer.texts_to_sequences([patient_problem])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequence)
    disease_index = np.argmax(prediction[0], axis=1)[0]
    prescription_index = np.argmax(prediction[1], axis=1)[0]
    disease_predicted = label_encoder_disease.inverse_transform([disease_index])[0]
    prescription_predicted = label_encoder_prescription.inverse_transform([prescription_index])[0]
    return disease_predicted, prescription_predicted

# Streamlit UI
st.set_page_config(page_title="Medical Diagnoser", page_icon="🩺", layout="wide")

# Background Image CSS
image_path = "C:\\Users\\91762\\Downloads\\7e1643586f4710913b96f1cd0c7516c9.jpg"
image_base64 = get_image_base64(image_path)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpeg;base64,{image_base64});
        background-size: cover;
    }}
    h1, h2, h3 {{
        color: black !important;
    }}
    .stTextInput > label {{
        color: White !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Medical Diagnoser")
st.write("Enter your symptoms to predict the disease and receive a suggested prescription.")

patient_problem = st.text_input("Describe your symptoms:")

if st.button("Diagnose"):
    if patient_problem:
        disease, prescription = make_prediction(patient_problem)
        st.subheader("Prediction Results")
        st.write(f"**Predicted Disease:** {disease}")
        st.write(f"**Suggested Prescription:** {prescription}")
    else:
        st.write("Please enter the symptoms.")
