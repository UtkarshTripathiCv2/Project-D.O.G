import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import base64


st.set_page_config(
    page_title="D.O.G. Vision System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg_from_url():
    # Using a high-quality, license-free image URL for the background
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1559493219-04a49c6c3fb8?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp > header {
        background-color: transparent;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

html, body, [class*="st-"] {
    font-family: 'Poppins', sans-serif;
}

.main-title {
    font-size: 3.5rem;
    font-weight: 600;
    color: #FFFFFF;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
    text-align: center;
    padding: 20px 0;
}

.sub-header {
    font-size: 1.2rem;
    color: #FFFFFF;
    text-align: center;
    text-shadow: 1px 1px 4px rgba(0,0,0,0.8);
    margin-bottom: 30px;
}

[data-testid="stSidebar"] {
    background-color: rgba(14, 54, 14, 0.8);
    backdrop-filter: blur(5px);
}

[data-testid="stSidebar"] h2 {
    color: #9AE6B4; /* Light Green */
}

.stButton>button {
    background-color: #38A169; /* Green */
    color: white;
    border-radius: 12px;
    border: 2px solid #2F855A; /* Darker Green */
    padding: 10px 20px;
    transition: all 0.3s ease-in-out;
}

.stButton>button:hover {
    background-color: #2F855A;
    border-color: #9AE6B4;
}

.results-container {
    background-color: rgba(255, 255, 255, 0.9);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}
</style>
""", unsafe_allow_html=True)

set_bg_from_url()



@st.cache_resource
def load_model(model_path):
    """Loads the YOLOv8 model from the specified path."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Drawing/Annotation Logic ---
def annotate_frame(frame, model, confidence_threshold):
    """Annotates a single frame with YOLOv8 detections."""
    np.random.seed(42)
    colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(model.names))]
    
    results = model(frame, verbose=False)
    annotated_frame = frame.copy()
    detection_count = 0
    detections_list = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            confidence = box.conf[0]
            
            if confidence > confidence_threshold:
                detection_count += 1
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                detections_list.append(f"{class_name} ({confidence:.2f})")
                color = colors[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)

                label = f'{class_name} {confidence:.2f}'
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_bg_x2 = x1 + text_width + 10
                label_bg_y2 = y1 + text_height + baseline + 10
                text_x = x1 + 5
                text_y = y1 + text_height + 5

                cv2.rectangle(annotated_frame, (x1, y1), (label_bg_x2, label_bg_y2), color, -1)
                cv2.putText(annotated_frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return annotated_frame, detection_count, detections_list

st.markdown('<p class="main-title">D.O.G. Vision System 🤖</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Agricultural Monitoring</p>', unsafe_allow_html=True)

model = load_model("DOG.pt")

if model:
    st.sidebar.title("Controls")
    st.sidebar.markdown("---")
    source_option = st.sidebar.radio("Input Source", ["🖼️ Upload an Image", "📹 Live Webcam Feed"], index=0)
    
    st.sidebar.header("Detection Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    if source_option == "🖼️ Upload an Image":
        uploaded_file = st.file_uploader("Select an image from your device", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            with st.spinner("🔍 Analyzing your field..."):
                annotated_frame, detection_count, detections_list = annotate_frame(frame, model, confidence_threshold)
            
            st.success(f"✅ Analysis complete! Found {detection_count} objects.")
            
            with st.container():
                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Original Image")
                    st.image(frame, channels="BGR", use_column_width=True)
                with col2:
                    st.header("Detection Results")
                    st.image(annotated_frame, channels="BGR", use_column_width=True)
                
                if detections_list:
                    st.header("Detected Objects:")
                    st.info(" | ".join(detections_list))
                st.markdown('</div>', unsafe_allow_html=True)


    elif source_option == "📹 Live Webcam Feed":
        st.header("Webcam Live Feed")
        st.write("Click 'START' to activate your camera. Grant permissions if prompted by your browser.")

        class VideoTransformer(VideoTransformerBase):
            def __init__(self):
                self.model = load_model("DOG.pt")
                self.confidence_threshold = confidence_threshold

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                if self.model:
                    # The confidence needs to be updated if the slider changes
                    # A more advanced implementation would use session state here
                    annotated_frame, _, _ = annotate_frame(img, self.model, confidence_threshold)
                    return annotated_frame
                return img

        webrtc_streamer(key="webcam", video_transformer_factory=VideoTransformer, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

else:
    st.error("Model file 'DOG.pt' not found. Please place it in the same directory as this script.")

