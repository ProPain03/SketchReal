import streamlit as st
import base64
import os

def load_css():
    """
    Load custom CSS styles for the app
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the CSS file
    css_file = os.path.join(current_dir, "style.css")
    
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback inline CSS if file is not found
        st.warning(f"CSS file not found at: {css_file}")
        st.markdown("""
        <style>
            /* Fallback CSS */
            h1 {
                color: #4285F4;
                font-weight: 700;
            }
            
            h3 {
                color: #4B4B4B;
                font-weight: 500;
            }
            
            .stButton button {
                background-color: #4285F4;
                color: white;
                font-weight: bold;
                border-radius: 6px;
            }
            
            .stButton button:hover {
                background-color: #3367D6;
            }
            
            .download-button {
                background-color: #34A853;
                border: none;
                color: white;
                padding: 12px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 10px 0;
                cursor: pointer;
                border-radius: 6px;
                font-weight: bold;
            }
            
            .download-button:hover {
                background-color: #2E8B57;
            }
            
            .stTabs [aria-selected="true"] {
                background-color: #4285F4 !important;
                color: white !important;
            }
            
            .stImage img {
                border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }
        </style>
        """, unsafe_allow_html=True)

def create_download_link(code, filename):
    """
    Create a download link for the generated HTML code
    
    Args:
        code: The HTML code as a string
        filename: The name of the file to download
        
    Returns:
        HTML string containing download link
    """
    b64 = base64.b64encode(code.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}" class="download-button">💾 Download HTML</a>'
    return href