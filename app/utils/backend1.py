import google.generativeai as genai
from PIL import Image
import io
import re

def call(image_bytes: bytes, prompt: str, service_key: str) -> str:
    """
    Process image with AI vision model to generate HTML/CSS code
    
    Args:
        image_bytes: The image file as bytes
        prompt: The text prompt to send with the image
        service_key: Service authentication key
        
    Returns:
        The generated HTML/CSS code as a string
    """
    # Configure the service
    genai.configure(api_key=service_key)
    
    # Set up the AI model
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Process the image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Generate content
    response = model.generate_content([img, prompt])
    
    # Extract code from response (if wrapped in markdown code blocks)
    code = response.text
    
    # Clean up the response if it contains markdown code blocks
    code_block_pattern = r"```(?:html)?(.*?)```"
    code_blocks = re.findall(code_block_pattern, code, re.DOTALL)
    
    if code_blocks:
        # Join all code blocks if multiple are found
        return "\n\n".join(block.strip() for block in code_blocks)
    
    return code