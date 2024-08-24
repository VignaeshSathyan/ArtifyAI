import streamlit as st
from PIL import Image, UnidentifiedImageError
import io
import logging
from prompt_generator import PromptGenerator
from image_generator import ImageGenerator
from img2img import Image2Image
import numpy as np
import torch

# Explicitly configure logging only once
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

# Ensure deprecation warnings are addressed
torch.utils._pytree.register_pytree_node = torch.utils._pytree._register_pytree_node

if 'llm_prompt' not in st.session_state:
    st.session_state['llm_prompt'] = None

st.set_page_config(
    page_title="RagArt",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨RagProject")

with st.sidebar:
    image = st.file_uploader(label="Upload an image",
                             label_visibility="collapsed",
                             type=["png", "jpg", "jpeg"], 
                             accept_multiple_files=False, 
                             help="Upload an image to use HuggingFace Img2Img model")
    st.write("")
    st.write("")
    st.write("Vignaesh Sathyan")
    st.write("[GitHub](https://github.com/VignaeshSathyan)")
    st.write("[LinkedIn](https://www.linkedin.com/in/vignaesh-sathyan-231275267?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)")

col1, col2 = st.columns(2, gap="large")

def load_image(image_file):
    try:
        img = Image.open(image_file)
        img.verify()  # Verify that it is, in fact, an image
        img = Image.open(image_file)  # Reopen for actual use
        return img
    except UnidentifiedImageError:
        st.error("Failed to identify the uploaded image. Please upload a valid image file.")
        logger.error("Failed to identify the uploaded image.")
        return None
    except Exception as e:
        st.error(f"Error loading image: {e}")
        logger.error(f"Error loading image: {e}")
        return None

with col1:
    if image:
        st.write("Image uploaded")
        img = load_image(image)
        if img:
            st.image(img, caption=image.name, use_column_width=True)
            st.write(f"Image size: {img.size}, mode: {img.mode}")
            logger.debug(f"Uploaded image name: {image.name}, type: {type(image)}")
        else:
            st.error("Failed to open the uploaded image.")
    
    user_prompt = st.text_input(label="Your prompt idea here")
    if st.button(label="Generate"):
        if not image:
            st.error("Please upload an image first.")
        else:
            prompt_generator = PromptGenerator()
            llm_prompt = prompt_generator.get_response(user_prompt)
            st.session_state['llm_prompt'] = llm_prompt
            logger.debug(f"Generated prompt: {llm_prompt}")

if 'llm_prompt' in st.session_state and st.session_state['llm_prompt'] is not None:
    with col2:
        try:
            if img:
                image_generator = Image2Image(img=img)
            else:
                image_generator = ImageGenerator()
            with st.spinner("Generating image..."):
                prompt = st.session_state['llm_prompt']
                st.write(prompt)
                generated_image = image_generator.generate(prompt=prompt)
                
                if generated_image is None:
                    st.error("Failed to generate image.")
                else:
                    # Validate generated image
                    image_array = np.array(generated_image)
                    if np.any(np.isnan(image_array)):
                        st.error("Generated image contains NaN values.")
                        logger.error("Generated image contains NaN values.")
                    elif image_array.min() == image_array.max() == 0:
                        st.error("Generated image is blank.")
                        logger.error("Generated image is blank.")
                    else:
                        img_bytes = io.BytesIO()
                        generated_image.save(img_bytes, format='PNG')
                        img_bytes = img_bytes.getvalue()
                        
                        st.image(generated_image, width=500)
                        st.download_button(label="Download", data=img_bytes, 
                                           file_name="image.png", mime='image/png')
                        logger.debug("Image displayed and download button provided.")
                        
                        # Display the input and output images for debugging
                        st.image(img, caption="Input Image", use_column_width=True)
                        st.image(generated_image, caption="Generated Image", use_column_width=True)
        
        except ValueError as ve:
            st.error(f"ValueError: {ve}")
            logger.error(f"ValueError: {ve}")
        except Exception as e:
            st.error(f"Error generating image: {e}")
            logger.error(f"Error generating image: {e}")
