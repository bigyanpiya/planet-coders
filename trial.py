import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import google.generativeai as genai
import json
import re
from tensorflow.keras.models import load_model
API_KEY = "AIzaSyDJ6S_VtAI6GM5OrSjtGufM4q8OeoQZROc"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Load TFLite model and labels
interpreter = tf.lite.Interpreter(model_path="C:/Users/Lenovo/OneDrive/Desktop/stremlit/model_unquant.tflite")
interpreter.allocate_tensors()
labels = ["Toyota Innova", "Tata Safari", "Swift", "Scorpio", "Creta"]

# Function to preprocess the image
# Function to preprocess the image
def preprocess_image(image):
    # Resize image to the required input shape of the model
    image = image.resize((224, 224))
    # Convert image to numpy array, normalize pixel values, and convert to FLOAT32
    image = np.array(image, dtype=np.float32) / 255.0
    # Expand dimensions to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    return image


# Function to perform inference
def predict_vehicle(image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the image
    input_data = preprocess_image(image)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class
    predicted_class = np.argmax(output_data)

    return labels[predicted_class]

# Streamlit app
def main():
    st.title("Vehicle Evaluation")

    uploaded_image = st.file_uploader("Upload an image of a vehicle", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Perform inference
        vehicle_class = predict_vehicle(image)

        st.write(f"Predicted Vehicle: {vehicle_class}")

        response = model.generate_content("In horizontal table format, give me only one data of engine_size, cylinders, Transmission, Fuel_type, Fuel consumption city (L/100)km, Fuel consumption Hwy (L/100)km, Fuel consumption comb(L/100)km, Fuel consumption comb(mpg) of " + vehicle_class)
        st.write(response.text)

        # Regular expression pattern to extract numbers
        pattern = r'\d+\.?\d*'

        # Extracting numbers from the table data
        numbers = re.findall(pattern, response.text)

        # Converting numbers to float if they contain a dot, otherwise to integer
        numbers = [float(num) if '.' in num else int(num) for num in numbers]

        numbers=numbers[3:]
        from sklearn.preprocessing import MinMaxScaler


        # Create MinMaxScaler object
        scaler = MinMaxScaler()

        # Fit the scaler to your data and transform it
        numbers=np.array(numbers).reshape(-1,1)
        normalized_numbers = scaler.fit_transform(numbers)

        # Convert the numpy array back to a Python list
        normalized_numbers_list = normalized_numbers.flatten().tolist()

        print(normalized_numbers_list)
        from tensorflow.keras.models import load_model
  

        model1 = load_model('car.h5')

        predictions = model1.predict([normalized_numbers_list])
        

        st.write(f"CO2 Emission Predicted Value: {predictions*100}%")


        # response_arko = model.generate_content("In array, give me engine_size, cylinders, Transmission, Fuel_type, Fuel consumption city (L/100)km, Fuel consumption Hwy (L/100)km, Fuel consumption comb(L/100)km, Fuel consumption comb(mpg) of " + vehicle_class) 
        # print(response_arko.text)

        # data = json.loads(response_arko.text)

        # print(data)
        # def extract_numerical_values(string):
        #     numerical_values = re.findall(r'\d+\.\d+', string)
        #     return [float(value) for value in numerical_values]

        # # Extract numerical values from the JSON data
        # numerical_values = []
        # for value in data.values():
        #     if isinstance(value, str):
        #         numerical_values.extend(extract_numerical_values(value))

        # st.write(numerical_values)
        response = model.generate_content("In vertical table format give me other important datas and information like sound prollution it creates etc about "+vehicle_class)
        st.write(response.text)

if __name__ == "__main__":
    main()