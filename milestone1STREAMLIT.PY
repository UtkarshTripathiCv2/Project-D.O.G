import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- Page Configuration ---
st.set_page_config(
    page_title="D.O.G. Vision System | AI Monitoring",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Styling (CSS) ---
st.markdown("""
<style>
/* Import Google Fonts for a premium feel */
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&family=Lato:wght@400;700&display=swap');

/* Main App Styling */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1495534027489-3543734d35e1?q=80&w=1932&auto-format&fit=crop");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Hide Streamlit Header/Footer */
.stApp > header {
    background-color: transparent;
}
#MainMenu, footer {
    visibility: hidden;
}

/* --- Typography --- */
h1, h2, h3 {
    font-family: 'Montserrat', sans-serif;
    color: #FFFFFF;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.5);
}

p, .stMarkdown, .st-emotion-cache-1g8sfsg {
    font-family: 'Lato', sans-serif;
    color: #E0E0E0;
}

/* --- Main Title & Header --- */
.main-title {
    font-size: 3.8rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0;
}

.sub-header {
    font-size: 1.5rem;
    font-weight: 400;
    text-align: center;
    color: #FFCA28; /* Light Amber for subtitle */
    margin-bottom: 40px;
}

/* --- Glassmorphism Containers --- */
.glass-container {
    background: rgba(10, 25, 10, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    padding: 2rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

/* --- Sidebar Styling --- */
[data-testid="stSidebar"] {
    background: rgba(20, 20, 20, 0.8); /* Darker sidebar */
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}
[data-testid="stSidebar"] h2 {
    color: #FFC107; /* Amber */
}

/* --- Custom Button Styling --- */
.stButton>button {
    font-family: 'Lato', sans-serif;
    background-color: #FFB300; /* Amber */
    color: #111111; /* Dark Text */
    border-radius: 10px;
    border: 1px solid #FFA000; /* Darker Amber */
    padding: 10px 24px;
    font-weight: 700;
    transition: all 0.3s ease-in-out;
}
.stButton>button:hover {
    background-color: #FFA000;
    border-color: #FFCA28;
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# --- Caching and Model Loading ---
@st.cache_resource
def load_model(model_path):
    """Loads and caches the YOLOv8 model."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Annotation Logic ---
def annotate_frame(frame, model, confidence_threshold):
    """Annotates a frame with refined YOLOv8 detections using a single accent color."""
    results = model(frame, verbose=False)
    annotated_frame = frame.copy()
    detections_list = []
    
    # A single, premium amber color for all detections
    detection_color = (0, 191, 255) # BGR for Amber/Gold

    for r in results:
        for box in r.boxes:
            confidence = box.conf[0]
            if confidence > confidence_threshold:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                detections_list.append((class_name, float(confidence)))

                # --- Draw cornered box ---
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                l = int(min(x2 - x1, y2 - y1) * 0.2)
                t = 2
                cv2.line(annotated_frame, (x1, y1), (x1 + l, y1), detection_color, t); cv2.line(annotated_frame, (x1, y1), (x1, y1 + l), detection_color, t)
                cv2.line(annotated_frame, (x2, y1), (x2 - l, y1), detection_color, t); cv2.line(annotated_frame, (x2, y1), (x2, y1 + l), detection_color, t)
                cv2.line(annotated_frame, (x1, y2), (x1 + l, y2), detection_color, t); cv2.line(annotated_frame, (x1, y2), (x1, y2 - l), detection_color, t)
                cv2.line(annotated_frame, (x2, y2), (x2 - l, y2), detection_color, t); cv2.line(annotated_frame, (x2, y2), (x2, y2 - l), detection_color, t)

                # --- Draw refined label ---
                label = f'{class_name} {confidence:.2f}'
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = y1 - 10 if y1 - 10 > h else y1 + h + 10
                cv2.rectangle(annotated_frame, (x1, label_y - h - 5), (x1 + w, label_y + 5), detection_color, -1)
                cv2.putText(annotated_frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2) # Dark text for contrast

    return annotated_frame, detections_list

# --- UI Components ---
def display_image_uploader(model):
    """Component for handling image uploads and displaying results."""
    st.header("Upload an Image for Analysis")
    uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        with st.spinner("🛰️ Analyzing your field..."):
            annotated_frame, detections = annotate_frame(frame, model, st.session_state.confidence)
        
        st.success(f"✅ Analysis complete! Found {len(detections)} objects.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(frame, channels="BGR", use_container_width=True)
        with col2:
            st.subheader("Detection Results")
            st.image(annotated_frame, channels="BGR", use_container_width=True)
            
        st.subheader("Detected Objects:")
        if detections:
            # Display detections in a more structured and clean way
            for item, conf in sorted(detections, key=lambda x: x[1], reverse=True):
                st.markdown(f"- **{item}** (Confidence: `{conf:.2f}`)")
        else:
            st.info("No objects were detected above the current confidence threshold.")

def display_video_uploader(model):
    """Component for handling video uploads, processing, and displaying results."""
    st.header("Upload a Video for Analysis")
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"], label_visibility="collapsed")
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Placeholders for real-time updates
        st_frame = st.empty()
        progress_bar = st.progress(0)
        
        output_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_tfile.name, fourcc, fps, (width, height))

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame, _ = annotate_frame(frame, model, st.session_state.confidence)
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
            progress_bar.progress((i + 1) / frame_count)
            out.write(annotated_frame)

        cap.release()
        out.release()
        
        st.success("✅ Video analysis complete!")
        st.subheader("Download Processed Video")
        with open(output_tfile.name, "rb") as f:
            st.download_button(
                label="Download as MP4",
                data=f,
                file_name="detected_video.mp4",
                mime="video/mp4"
            )

def display_webcam_feed(model):
    """Component for handling the live webcam feed."""
    st.header("Live Webcam Feed")
    st.info("Click 'START' to activate your camera. The confidence slider on the left works in real-time!")

    class VideoTransformer(VideoTransformerBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            annotated_frame, _ = annotate_frame(img, model, st.session_state.confidence)
            return annotated_frame
            
    webrtc_streamer(
        key="webcam", 
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

# --- Main Application ---
def main():
    st.markdown('<p class="main-title">D.O.G. Vision System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Agricultural Monitoring</p>', unsafe_allow_html=True)

    model = load_model("DOG.pt")
    if not model:
        st.error("Model file 'DOG.pt' not found. Please ensure it is in the same directory.")
        return

    # --- Sidebar Controls ---
    st.sidebar.title("🌿 D.O.G. Vision")
    
    if 'confidence' not in st.session_state:
        st.session_state.confidence = 0.5
    
    st.session_state.confidence = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, st.session_state.confidence, 0.05
    )
    
    st.sidebar.markdown("---")
    source_option = st.sidebar.radio(
        "Input Source", 
        ["🖼️ Image Analysis", "📹 Video Analysis", "🛰️ Live Monitoring"],
        index=0
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        This YOLOv8n model was trained on a custom dataset of over 
        50,000 images for high-accuracy agricultural monitoring.
        """
    )

    # --- Main Content Area ---
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    if source_option == "🖼️ Image Analysis":
        display_image_uploader(model)
    elif source_option == "📹 Video Analysis":
        display_video_uploader(model)
    elif source_option == "🛰️ Live Monitoring":
        display_webcam_feed(model)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

