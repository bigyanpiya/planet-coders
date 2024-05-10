import streamlit as st
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Streamlit app
def main():
    st.title("Car Prediction App")

    # Create input fields for parameters
    engine_size = st.number_input("Engine Size")
    cylinders = st.number_input("Cylinders")
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    fuel_type = st.selectbox("Fuel Type", ["Regular", "Premium", "Diesel"])
    city_consumption = st.number_input("Fuel Consumption City (L/100 km)")
    hwy_consumption = st.number_input("Fuel Consumption Hwy (L/100 km)")
    comb_consumption = st.number_input("Fuel Consumption Comb (L/100 km)")
    comb_mpg = st.number_input("Fuel Consumption Comb (mpg)")

    # When submit button is clicked
    if st.button("Predict"):
        # Convert categorical variables to numerical
        transmission = 0 if transmission == "Manual" else 1
        fuel_type = {"Regular": 0, "Premium": 1, "Diesel": 2}[fuel_type]

        # Preprocess input data
        input_data = np.array([engine_size, cylinders, transmission, fuel_type, city_consumption, hwy_consumption, comb_consumption, comb_mpg], dtype=np.float32)
        
        # Make prediction
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        st.write("Predicted Value:", prediction)

if __name__ == "__main__":
    main()
