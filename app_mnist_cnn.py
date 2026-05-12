import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import keras

# Load trained CNN model
@st.cache_resource
def load_model():
 
    return tf.keras.models.load_model("cnn_mnist_model.keras")
#    r"C:\Python\Python_kurs\AI bok\cnn_mnist_model.keras"
#)

model = load_model()

st.title("MNIST Digit Classifier (CNN)")
st.write("Draw a digit below and click Predict")

# Canvas
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:

        # Convert canvas to PIL image
        img = Image.fromarray(
            canvas_result.image_data.astype("uint8"),
            mode="RGBA"
        )

        # Convert to grayscale
        img = img.convert("L")

        # Resize to 28x28
        img = img.resize((28, 28))

        # Convert to numpy
        img_array = np.array(img).astype("float32")

        # Normalize (IMPORTANT for CNN)
        img_array = img_array / 255.0

        # Reshape for CNN input
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        st.success(f"Predicted Digit: {predicted_digit}")

        # Optional: Show processed image
        st.image(img, caption="Processed 28x28 Image", width=150)