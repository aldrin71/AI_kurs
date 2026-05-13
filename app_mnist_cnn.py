import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# ---------------------------------------------------
# Load CNN model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        r"C:\Python\Python_kurs\Kunskapskontroll_AI\cnn_mnist_model.keras"
    )

model = load_model()

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.title("MNIST Digit Classifier")
st.write("Draw a digit below and click Predict")

# ---------------------------------------------------
# Canvas
# ---------------------------------------------------
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ---------------------------------------------------
# MNIST-style preprocessing
# ---------------------------------------------------
def preprocess_image(image):

    # Convert to grayscale
    image = image.convert("L")

    # Convert to numpy
    img_array = np.array(image)

    # Threshold
    img_array = np.where(img_array > 50, 255, 0)

    # Convert back to PIL
    image = Image.fromarray(img_array.astype(np.uint8))

    # Crop bounding box
    bbox = image.getbbox()

    if bbox is not None:
        image = image.crop(bbox)

    # Resize while preserving aspect ratio
    image.thumbnail((20, 20))

    # Create new 28x28 black image
    new_img = Image.new("L", (28, 28), 0)

    # Center digit
    paste_x = (28 - image.width) // 2
    paste_y = (28 - image.height) // 2

    new_img.paste(image, (paste_x, paste_y))

    # Convert to array
    img_array = np.array(new_img).astype("float32")

    # Normalize
    img_array = img_array / 255.0

    # Reshape for CNN
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array, new_img

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("Predict"):

    if canvas_result.image_data is not None:

        # Convert canvas to image
        img = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        )

        # Preprocess
        processed_array, processed_img = preprocess_image(img)

        # Predict
        prediction = model.predict(processed_array)

        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)

        # ---------------------------------------------------
        # Results
        # ---------------------------------------------------
        st.success(f"Predicted Digit: {predicted_digit}")
        st.write(f"Confidence: {confidence:.2%}")

        # Show processed image
        st.subheader("Processed Image (What the CNN sees)")
        st.image(processed_img, width=150)

        # Probabilities
        st.subheader("Prediction Probabilities")

        for i, prob in enumerate(prediction[0]):
            st.write(f"Digit {i}: {prob:.4f}")
            st.progress(float(prob))