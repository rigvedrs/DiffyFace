"""
Streamlit Web Interface for DiffyFace Image Generation

This application provides a user-friendly interface for generating face images
from text prompts using the DiffyFace model.
"""

import io
import os
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Generation.model import DiffyFaceModel


# Page configuration
st.set_page_config(
    page_title="DiffyFace - AI Face Generator",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #764ba2;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """
    Load the DiffyFace model with caching to avoid reloading on every interaction.
    
    Returns:
        DiffyFaceModel instance
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        st.error("❌ CUDA not available. DiffyFace requires a GPU to run.")
        st.info("Please run this on a machine with CUDA support.")
        return None
    
    try:
        model = DiffyFaceModel(device="cuda")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🎭 DiffyFace - AI Face Generator</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Device selection
        device_options = ["Auto-detect", "CPU", "CUDA"]
        selected_device = st.selectbox(
            "Device",
            device_options,
            index=0 if not torch.cuda.is_available() else 0
        )
        
        # Generation parameters
        st.header("🎨 Generation Parameters")
        
        num_steps = st.slider(
            "Number of Inference Steps",
            min_value=20,
            max_value=100,
            value=50,
            step=5,
            help="More steps = higher quality but slower generation"
        )
        
        guidance_scale = st.slider(
            "Guidance Scale",
            min_value=1.0,
            max_value=20.0,
            value=7.5,
            step=0.5,
            help="How closely to follow the prompt"
        )
        
        seed = st.number_input(
            "Random Seed (optional)",
            min_value=0,
            max_value=2147483647,
            value=None,
            help="Leave empty for random generation, or set a specific seed for reproducibility"
        )
        
        seed = int(seed) if seed is not None else None
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Enter Prompt")
        
        # Prompt input
        prompt = st.text_area(
            "Describe the face you want to generate:",
            height=100,
            placeholder="Example: A happy 25 year old male with blond hair and a french beard smiles with visible teeth.",
            help="Be descriptive about age, gender, hair, facial features, expression, etc."
        )
        
        # Negative prompt (optional)
        negative_prompt = st.text_area(
            "Negative Prompt (optional):",
            height=60,
            placeholder="Example: blurry, distorted, low quality",
            help="Describe what you don't want in the image"
        )
        
        # Generate button
        generate_button = st.button("🚀 Generate Image", type="primary", use_container_width=True)
    
    with col2:
        st.header("🖼️ Generated Image")
        
        # Initialize session state for image
        if "generated_image" not in st.session_state:
            st.session_state.generated_image = None
        if "last_prompt" not in st.session_state:
            st.session_state.last_prompt = ""
        
        # Display image if available
        if st.session_state.generated_image is not None:
            st.image(st.session_state.generated_image, use_container_width=True)
            
            # Save button
            if st.session_state.generated_image is not None:
                save_dir = PROJECT_ROOT / "Images" / "Saved_images"
                save_dir.mkdir(parents=True, exist_ok=True)
                
                filename = DiffyFaceModel._prompt_to_filename(st.session_state.last_prompt)
                save_path = save_dir / filename
                
                col_save1, col_save2 = st.columns([1, 1])
                with col_save1:
                    if st.button("💾 Save Image", use_container_width=True):
                        try:
                            st.session_state.generated_image.save(str(save_path))
                            st.success(f"✅ Image saved to:\n`{save_path}`")
                        except Exception as e:
                            st.error(f"Error saving image: {e}")
                
                with col_save2:
                    # Download button - convert PIL Image to bytes
                    img_bytes = io.BytesIO()
                    st.session_state.generated_image.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download Image",
                        data=img_bytes.getvalue(),
                        file_name=filename,
                        mime="image/png",
                        use_container_width=True
                    )
        else:
            st.info("Enter a prompt and click 'Generate Image' to create a face!")
    
    # Generate image when button is clicked
    if generate_button:
        if not prompt.strip():
            st.warning("⚠️ Please enter a prompt to generate an image.")
        else:
            # Show loading spinner
            with st.spinner("🎨 Generating your face image... This may take a minute."):
                try:
                    # Load model
                    model = load_model()
                    
                    if model is None:
                        st.error("❌ Failed to load model. Please check the error messages above.")
                    else:
                        # Generate image
                        image = model.generate(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=num_steps,
                            seed=seed
                        )
                        
                        # Store in session state
                        st.session_state.generated_image = image
                        st.session_state.last_prompt = prompt
                        
                        # Rerun to update display
                        st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error generating image: {e}")
                    st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>DiffyFace - AI-Powered Face Generation using Stable Diffusion with LoRA</p>
            <p>Built with ❤️ using Streamlit and Diffusers</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
