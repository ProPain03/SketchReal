import streamlit as st
import os
import base64
from PIL import Image
import io

from utils.gemini_api import call
from utils.ui_helpers import load_css, create_download_link

# Page configuration
st.set_page_config(
    page_title="SketchReal",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
load_css()

# App header with gradient background
st.markdown('<div class="main-header-container">', unsafe_allow_html=True)
st.markdown("# 🎨 SketchReal")
st.markdown("### Transform UI sketches into HTML/CSS code")
st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = ""
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False

# Sidebar controls
with st.sidebar:
    st.image("https://storage.googleapis.com/gweb-uniblog-publish-prod/images/gemini_1.max-1000x1000.png", width=100)
    st.markdown("## 🛠️ Settings")
    
    # Allow customizing the prompt
    st.markdown("### Customize Prompt")
    default_prompt = """This is a sketch or screenshot of a user interface. Generate the corresponding HTML and CSS to replicate the layout.
Use semantic tags like <button>, <input>, <div>, and modern styling practices like CSS Flexbox or Grid.
Make the design responsive and visually appealing.
Add realistic content (placeholder text, button labels) based on the UI context.
Output only the complete HTML and CSS code, no explanation or markdown formatting."""

    prompt = st.text_area("Prompt sent with image", 
                          value=default_prompt, 
                          height=200)
    
    # Check if service key exists in secrets
    if 'GEMINI_API_KEY' in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        st.warning("⚠️ Service key not found in configuration")
        api_key = st.text_input("Enter your service key", type="password")
    
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. Upload a UI sketch or screenshot
    2. The image is processed using AI technology
    3. System generates corresponding HTML/CSS
    4. Review and download the generated code
    """)

# Main content in two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📤 Upload UI Design")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded UI Design", use_container_width=True)
        st.session_state.image_uploaded = True
        
        # Generate button
        if st.button("🚀 Generate Code"):
            with st.spinner("🧠 Processing your image"):
                try:
                    # Get image bytes
                    img_bytes = uploaded_file.getvalue()
                    
                    # Process image and generate code
                    generated_code = call(img_bytes, prompt, api_key)
                    st.session_state.generated_code = generated_code
                    st.success("✅ Code generated successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with col2:
    st.markdown("## 💻 Generated Code")
    
    # Display generated code and download button if available
    if st.session_state.generated_code:
        # Create tabs for code and preview
        tab1, tab2 = st.tabs(["📝 Code", "👁️ Preview"])
        
        with tab1:
            st.code(st.session_state.generated_code, language="html")
            st.markdown(create_download_link(st.session_state.generated_code, "ui_design.html"), unsafe_allow_html=True)
            
        with tab2:
            st.components.v1.html(st.session_state.generated_code, height=600, scrolling=True)
    else:
        # Placeholder before generation
        if st.session_state.image_uploaded:
            st.info("👈 Click 'Generate Code' to transform your design")
        else:
            st.info("👈 Upload an image to get started")

# Footer
st.markdown("---")
st.markdown("<footer>Created with Streamlit</footer>", unsafe_allow_html=True)