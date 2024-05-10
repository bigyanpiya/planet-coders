import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import google.generativeai as genai
import re

API_KEY = "AIzaSyDJ6S_VtAI6GM5OrSjtGufM4q8OeoQZROc"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Load TFLite model for vehicle classification
vehicle_interpreter = tf.lite.Interpreter(model_path="C:/Users/Lenovo/OneDrive/Desktop/stremlit/model_unquant.tflite")
vehicle_interpreter.allocate_tensors()
labels = ["Toyota Innova", "Tata Safari", "Swift", "Scorpio", "Creta"]

# Load TFLite model for carbon emission prediction
carbon_emission_interpreter = tf.lite.Interpreter(model_path="C:/Users/Lenovo/OneDrive/Desktop/stremlit/converted_model.tflite")
carbon_emission_interpreter.allocate_tensors()

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to preprocess vehicle data for carbon emission prediction
def preprocess_vehicle_data(vehicle_data):
    # Convert the vehicle_data to a numpy array
    vehicle_data = np.array(vehicle_data, dtype=np.float32)
    # Reshape the input data to match the expected shape
    vehicle_data = vehicle_data.reshape((1, -1))  # Reshape to a row vector
    return vehicle_data

# Function to perform inference for vehicle classification
def predict_vehicle(image):
    input_details = vehicle_interpreter.get_input_details()
    output_details = vehicle_interpreter.get_output_details()

    input_data = preprocess_image(image)
    vehicle_interpreter.set_tensor(input_details[0]['index'], input_data)
    vehicle_interpreter.invoke()
    output_data = vehicle_interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)

    return labels[predicted_class]

# Function to perform inference for carbon emission prediction
def predict_carbon_emission(vehicle_data):
    input_details = carbon_emission_interpreter.get_input_details()
    output_details = carbon_emission_interpreter.get_output_details()

    input_data = preprocess_vehicle_data(vehicle_data)
    carbon_emission_interpreter.set_tensor(input_details[0]['index'], input_data)
    carbon_emission_interpreter.invoke()
    carbon_emission = carbon_emission_interpreter.get_tensor(output_details[0]['index'])

    return carbon_emission

# Streamlit app
def main():
    st.title("Vehicle Classification and Carbon Emission Prediction")

    uploaded_image = st.file_uploader("Upload an image of a vehicle", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Perform inference for vehicle classification
        vehicle_class = predict_vehicle(image)
        st.write(f"Predicted Vehicle: {vehicle_class}")

        # Get vehicle details from Gemini API response
        response = model.generate_content("In horizontal table format, give me only one data of engine_size, cylinders, Transmission, Fuel_type, Fuel consumption city (L/100)km, Fuel consumption Hwy (L/100)km, Fuel consumption comb(L/100)km, Fuel consumption comb(mpg) of " + vehicle_class)
        vehicle_data = re.findall(r'\d+\.?\d*', response.text)

        # Perform inference for carbon emission prediction
        carbon_emission = predict_carbon_emission(vehicle_data)

        st.write(f"Predicted Carbon Emission: {carbon_emission[0]}")  # Assuming the output is a single value

if __name__ == "__main__":
    main()
