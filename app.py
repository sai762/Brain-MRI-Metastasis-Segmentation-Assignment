import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K

# Define custom metrics
def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_coefficients(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) + smooth)
    return dice

# Define the custom loss function
def dice_coefficients_loss(y_true, y_pred):
    return 1 - dice_coefficients(y_true, y_pred)

# Load your trained models (U-Net and Attention U-Net)
@st.cache_resource
def load_models():
    model_unet = tf.keras.models.load_model(
        'unet_plus_plus_final.h5',
        custom_objects={'iou': iou, 'dice_coefficients': dice_coefficients, 'dice_coefficients_loss': dice_coefficients_loss}
    )
    model_attention_unet = tf.keras.models.load_model(
        'attention_unet_final.h5',
        custom_objects={'iou': iou, 'dice_coefficients': dice_coefficients, 'dice_coefficients_loss': dice_coefficients_loss}
    )
    return model_unet, model_attention_unet

# Image preprocessing function
def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Display segmentation results
def display_results(image, mask_unet, mask_attention_unet):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(image, caption='Original MRI Image', use_column_width=True)
        
    with col2:
        st.image(mask_unet, caption='U-Net++ Segmentation', use_column_width=True)
        
    with col3:
        st.image(mask_attention_unet, caption='Attention U-Net Segmentation', use_column_width=True)

# Main UI function
def main():
    st.title("Brain MRI Metastasis Segmentation")

    # Load models
    model_unet, model_attention_unet = load_models()

    st.sidebar.title("Upload MRI Image")
    uploaded_file = st.sidebar.file_uploader("Choose a Brain MRI image...", type=["jpg", "png", "jpeg", "tif"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess image for the model
        preprocessed_image = preprocess_image(image)
        
        # Get predictions from both models
        mask_unet = model_unet.predict(preprocessed_image)
        mask_attention_unet = model_attention_unet.predict(preprocessed_image)
        
        # Post-process the masks (optional, depending on your model output)
        mask_unet = np.squeeze(mask_unet)  # Remove batch dimension
        mask_attention_unet = np.squeeze(mask_attention_unet)  # Remove batch dimension
        
        # Resize masks back to original image size
        mask_unet = cv2.resize(mask_unet, (image.width, image.height))
        mask_attention_unet = cv2.resize(mask_attention_unet, (image.width, image.height))
        
        # Display the original image and segmentation results
        display_results(image, mask_unet, mask_attention_unet)

if __name__ == "__main__":
    main()
