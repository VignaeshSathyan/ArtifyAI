from PIL import Image
import io
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)

class Image2Image:
    def __init__(self, img, model_id="timbrooks/instruct-pix2pix"):
        self.img = img
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
    
    def generate(self, prompt: str):
        logging.debug("Starting image-to-image generation.")
        if not isinstance(self.img, Image.Image):
            try:
                # Open the image using PIL
                pil_image = Image.open(io.BytesIO(self.img.read()))
                logging.debug("Image loaded successfully using PIL.")
            except Exception as e:
                raise ValueError("Failed to open the image.") from e
        else:
            pil_image = self.img

        # Save the input image for debugging
        pil_image.save("input_image.png")

        output = self.pipe(prompt, image=pil_image, num_inference_steps=10, image_guidance_scale=1)
        if not output.images:
            raise ValueError("No images generated.")

        image = output.images[0]

        # Verify image integrity
        if image is None or image.mode != "RGB":
            raise ValueError("Generated image is invalid.")
        
        logging.debug("Generated image successfully.")
        
        # Save the generated image for debugging
        image.save("output_image.png")
        
        # Log the generated image stats
        np_image = np.array(image)
        logging.debug(f"Image stats - min: {np.min(np_image)}, max: {np.max(np_image)}, mean: {np.mean(np_image)}")
        
        return image
