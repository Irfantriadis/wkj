import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time

# Set page configuration
st.set_page_config(
    page_title="Herbal Plant Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and better styling
st.markdown("""
    <style>
    /* Main background and text colors */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #262730;
    }
    
    /* Header styling */
    .main .block-container {
        padding-top: 2rem;
    }
    
    h1, h2, h3, h4 {
        color: #FAFAFA;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background-color: #262730;
        border: 1px solid #0E1117;
        border-radius: 5px;
        padding: 1rem;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #00CA85;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 500;
    }
    
    .stButton button:hover {
        background-color: #00B377;
    }
    
    /* Results box styling */
    .prediction-box {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #404040;
    }
    
    /* Radio button styling */
    .stRadio > label {
        color: #FAFAFA;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #262730;
        color: #FAFAFA;
    }
    
    /* File uploader styling */
    .stUploadedFile {
        background-color: #262730 !important;
    }
    
    /* Image container styling */
    .image-container {
        background-color: #262730;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Center align file uploader */
    .css-1v0mbdj.e115fcil1 {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #262730;
        padding: 0.5rem;
        border-radius: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 5px;
        color: #FAFAFA;
        padding: 0 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00CA85 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Same model paths as before
MODEL_PATHS = {
    'leaf': {
        'model': "model daun/mobilenetv2_model.tflite",
        'labels': "model daun/labels.txt"
    },
    'fruit': {
        'model': "model buah/mobilenetV2_model.tflite",
        'labels': "model buah/labels.txt"
    },
    'rhizome': {
        'model': "model rimpang/mobilenetv2_model.tflite",
        'labels': "model rimpang/labels.txt"
    }
}

def load_model_and_labels(model_name):
    model_info = MODEL_PATHS[model_name]
    interpreter = tf.lite.Interpreter(model_path=model_info['model'])
    interpreter.allocate_tensors()
    with open(model_info['labels'], 'r') as f:
        labels = f.read().splitlines()
    return interpreter, labels

def preprocess_image(image):
    if isinstance(image, str):
        image = Image.open(image)
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict(image, model_name):
    interpreter, labels = load_model_and_labels(model_name)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    img = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)
    confidence = output_data[0][predicted_index]
    
    return labels[predicted_index], confidence

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.capture_frame = None
        self.should_capture = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.should_capture:
            self.capture_frame = img.copy()
            self.should_capture = False
        return img

def main():
    # Header with custom styling
    st.markdown("""
        <h1 style='text-align: center; color: #00CA85; margin-bottom: 2rem;'>
            🌿 Herbal Plant Classification
        </h1>
    """, unsafe_allow_html=True)
    
    # Main selection at the top
    col1, col2 = st.columns([2, 2])
    
    with col1:
        model_name = st.selectbox(
            "Select Plant Part to Classify",
            ['leaf', 'fruit', 'rhizome'],
            format_func=lambda x: x.capitalize()
        )
    
    with col2:
        input_method = st.radio(
            "Choose Input Method",
            ["Upload Image", "Use Camera"],
            horizontal=True
        )
    
    st.markdown("---")
    
    # Main content area
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Select image file",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            cols = st.columns([2, 2])
            
            with cols[0]:
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with cols[1]:
                with st.spinner("🔍 Analyzing image..."):
                    image = Image.open(uploaded_file)
                    label, confidence = predict(image, model_name)
                    
                    st.markdown("""
                        <div class='prediction-box'>
                            <h3 style='color: #00CA85; margin-bottom: 1rem;'>Results</h3>
                            <p style='font-size: 1.2rem; margin-bottom: 0.5rem;'>
                                <strong>Predicted Plant:</strong> {}</p>
                            <p style='font-size: 1.2rem;'>
                                <strong>Confidence:</strong> {:.2f}%</p>
                        </div>
                    """.format(label, confidence*100), unsafe_allow_html=True)
    
    else:  # Camera input
        st.markdown("<h3>📸 Live Camera Feed</h3>", unsafe_allow_html=True)
        webrtc_ctx = webrtc_streamer(
            key="herbal-plant-detection",
            video_transformer_factory=VideoTransformer,
            async_processing=True
        )
        
        if webrtc_ctx.video_transformer:
            if st.button("📸 Capture Image"):
                webrtc_ctx.video_transformer.should_capture = True
                time.sleep(0.5)
                
                if webrtc_ctx.video_transformer.capture_frame is not None:
                    captured_image = Image.fromarray(cv2.cvtColor(
                        webrtc_ctx.video_transformer.capture_frame,
                        cv2.COLOR_BGR2RGB
                    ))
                    
                    cols = st.columns([2, 2])
                    
                    with cols[0]:
                        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                        st.image(captured_image, caption="Captured Image", use_column_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with cols[1]:
                        with st.spinner("🔍 Analyzing image..."):
                            label, confidence = predict(captured_image, model_name)
                            
                            st.markdown("""
                                <div class='prediction-box'>
                                    <h3 style='color: #00CA85; margin-bottom: 1rem;'>Results</h3>
                                    <p style='font-size: 1.2rem; margin-bottom: 0.5rem;'>
                                        <strong>Predicted Plant:</strong> {}</p>
                                    <p style='font-size: 1.2rem;'>
                                        <strong>Confidence:</strong> {:.2f}%</p>
                                </div>
                            """.format(label, confidence*100), unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #808495; padding: 1rem;'>
            <p>Built with Streamlit • Powered by TensorFlow</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
