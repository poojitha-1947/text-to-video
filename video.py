import streamlit as st
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import cv2
import pyttsx3
import os

# Streamlit app header
st.title("Text-to-Video Generator")
st.write("Enter a text description, and the system will generate a video!")

# Load model and pipeline for text-to-image generation using diffusers library
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype='auto')
pipe.to("cuda")  # Move the pipeline to GPU if available

# Function to generate image from text
def generate_image_from_text(text):
    # Generate image using Stable Diffusion pipeline
    image = pipe(text).images[0]  # Get the first image generated
    return image

# Text-to-Speech function
engine = pyttsx3.init()

def text_to_speech(text, output_path="narration.mp3"):
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    return output_path

# Video generation function
def generate_video_from_texts(texts, output_video_path="output_video.mp4"):
    frames = []
    for text in texts:
        img = generate_image_from_text(text)
        frames.append(np.array(img))

    # Check if there are frames generated
    if not frames:
        st.error("No frames were generated. Please check the input texts.")
        return

    # Extract frame size from the first frame
    frame_height, frame_width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Set codec for video
    video_out = cv2.VideoWriter(output_video_path, fourcc, 1, (frame_width, frame_height))

    # Write frames to the video
    for frame in frames:
        video_out.write(frame)

    video_out.release()

# Streamlit UI Elements
input_texts = st.text_area("Enter text descriptions (separate by new lines)", height=200)
if input_texts:
    texts = input_texts.split("\n")
    st.write(f"Generating video for: {texts}")

    # Button to generate video
    if st.button("Generate Video"):
        with st.spinner('Generating images and video...'):
            # Generate video from texts
            video_path = "generated_video.mp4"
            generate_video_from_texts(texts, video_path)

            # Check if video is generated and display it
            if os.path.exists(video_path):
                st.video(video_path)

                # Optionally generate narration
                narration_path = text_to_speech(" ".join(texts))
                st.write("Narration generated. Download it here:")
                st.audio(narration_path, format="audio/mp3")
                
                # Provide a download link for the generated video
                st.write(f"Download the generated video:")
                st.download_button("Download Video", video_path)
            else:
                st.error("Video generation failed.")
