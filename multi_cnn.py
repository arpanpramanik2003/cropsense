import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('crop_condition_model.h5')
    return model

# Load crop encoder classes
def load_crop_classes():
    return np.load('crop_encoder_classes.npy', allow_pickle=True)

# Preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
    return image_array

# Grad-CAM function
def direct_gradcam_heatmap(model, img_array, last_conv_layer_name, output_name, pred_index=None):
    try:
        conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.get_layer(output_name).output]
        )
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, preds = grad_model(img_tensor)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None:
            raise ValueError(f"Gradients are None for {output_name}")
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy(), pred_index
    except Exception as e:
        st.error(f"Error in Grad-CAM for {output_name}: {e}")
        return None, None

# Display Grad-CAM
def display_gradcam(image, heatmap_crop, heatmap_condition, crop_class, condition_class, alpha=0.4):
    img = np.array(image.resize((224, 224)))
    fig = plt.figure(figsize=(12, 4), dpi=300)
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image', fontsize=12, fontweight='bold')
    plt.axis('off')

    # Crop Grad-CAM
    if heatmap_crop is not None:
        heatmap_crop = cv2.resize(heatmap_crop, (224, 224))
        heatmap_crop = np.uint8(255 * heatmap_crop)
        heatmap_crop = cv2.applyColorMap(heatmap_crop, cv2.COLORMAP_JET)
        heatmap_crop = cv2.applyColorMap(heatmap_crop, cv2.COLORMAP_INFERNO)
        superimposed_crop = heatmap_crop * alpha + img
        superimposed_crop = np.clip(superimposed_crop, 0, 255).astype(np.uint8)
        plt.subplot(1, 3, 2)
        plt.imshow(superimposed_crop)
        plt.title(f'Crop: {crop_class}', fontsize=12, fontweight='bold')
        plt.axis('off')
    
    # Condition Grad-CAM
    if heatmap_condition is not None:
        heatmap_condition = cv2.resize(heatmap_condition, (224, 224))
        heatmap_condition = np.uint8(255 * heatmap_condition)
        heatmap_condition = cv2.applyColorMap(heatmap_condition, cv2.COLORMAP_JET)

        superimposed_condition = heatmap_condition * alpha + img
        superimposed_condition = np.clip(superimposed_condition, 0, 255).astype(np.uint8)
        plt.subplot(1, 3, 3)
        plt.imshow(superimposed_condition)
        plt.title(f'Condition: {condition_class}', fontsize=12, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout(pad=1.0)
    return fig

# Main Streamlit app
st.title("🥔🍅Crop and Condition Classifier")
st.write("Upload an image to classify the crop type and its condition, with Grad-CAM visualizations.")

model = load_model()
crop_classes = load_crop_classes()
condition_classes = ['Bad', 'Good']

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess and predict
    image_array = preprocess_image(image)
    predictions = model.predict(image_array)

    # Crop prediction
    crop_pred = np.argmax(predictions[0], axis=1)[0]
    crop_confidence = np.max(predictions[0]) * 100
    crop_label = crop_classes[crop_pred]

    # Condition prediction
    condition_pred = np.argmax(predictions[1], axis=1)[0]
    condition_confidence = np.max(predictions[1]) * 100
    condition_label = condition_classes[condition_pred]
    
    # Display predictions
    st.write(f"**Crop Prediction**: {crop_label} ({crop_confidence:.2f}% confidence)")
    st.write(f"**Condition Prediction**: {condition_label} ({condition_confidence:.2f}% confidence)")
    
    # Generate and display Grad-CAM
    st.subheader("Grad-CAM Explainability")
    heatmap_crop, crop_pred_index = direct_gradcam_heatmap(model, image_array, 'conv2d_2', 'crop')
    heatmap_condition, condition_pred_index = direct_gradcam_heatmap(model, image_array, 'conv2d_2', 'condition')
    if heatmap_crop is not None and heatmap_condition is not None:
        fig = display_gradcam(image, heatmap_crop, heatmap_condition, crop_label, condition_label)
        st.pyplot(fig)
