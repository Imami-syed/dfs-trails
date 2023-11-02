
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('mcc2.h5')

# Streamlit page title and description
st.title("Image Classification App")
st.write("Upload an image, and we'll predict the class.")

# Create a file uploader widget
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Make a prediction when the "Predict" button is clicked
    if st.button("Predict"):
        try:
            # Load and preprocess the uploaded image
            img = image.load_img(uploaded_image, target_size=(224, 224))  # Adjust target_size to match your model's input size
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
            img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

            # Make predictions using the loaded model
            predictions = model.predict(img_array)

            # Decode the predictions if your model has categorical labels (e.g., using ImageDataGenerator with class_mode='categorical')
            # If you trained the model with a different label encoding, adjust this part accordingly
            class_labels = ['all_pre', 'all_early', 'oral_scc']  # Replace with your class labels
            predicted_class = class_labels[np.argmax(predictions)]

            # Display the prediction
            st.write(f"Prediction: {predicted_class}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")