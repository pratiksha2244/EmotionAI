import streamlit as st
import matplotlib.pyplot as plt

import cv2
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
from tqdm.notebook import tqdm

import torch
from facenet_pytorch import (MTCNN)

from transformers import (AutoFeatureExtractor,
                          AutoModelForImageClassification,
                          AutoConfig)

from PIL import Image
from moviepy.editor import VideoFileClip, ImageSequenceClip

## Import modules
import av_recorder
from av_seprator import seperator
#from audio_app import inference
import video_app as vp


os.environ['XDG_CACHE_HOME'] = '/home/msds2023/jlegara/.cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/msds2023/jlegara/.cache'


## Input file with audio
#input_file = 'videos/output_with_audio.mp4'



scene = 'videos/output_with_audio.mp4'

def video_app():
    ## Input load video
    
    combined_images, _, confusion_matrix_image = vp.proba(scene)
    _, vid_fps = vp.video_capture(scene)
    confusion_matrix_image = None  # Replace None with the actual confusion matrix image
    
    return combined_images, vid_fps, confusion_matrix_image


#to display graph
def emotion_probability_plot(all_class_probabilities):
    # Define colors for each emotion
    colors = {
        "angry": "red",
        "disgust": "green",
        "fear": "gray",
        "happy": "yellow",
        "neutral": "purple",
        "sad": "blue",
        "surprise": "orange"
    }
    #all_class_probabilities.shape

   # Check if all_class_probabilities is empty or None
    if all_class_probabilities is None or len(all_class_probabilities) == 0:
        st.warning("No data available for plotting.")
        return
    
    # Check if all_class_probabilities contains arrays
    if isinstance(all_class_probabilities[0], np.ndarray):
        # Convert list of class probabilities into a NumPy array
        all_class_probabilities_array = np.array(all_class_probabilities)
        # Check if the array has a valid shape
        if len(all_class_probabilities_array.shape) != 3:
            st.error("Invalid shape for class probabilities array.")
            return
        
        # Calculate mean probability for each emotion across all frames
        mean_probabilities = np.mean(all_class_probabilities_array, axis=(0, 1))
    else:
        # If all_class_probabilities contains a single array (single frame)
        mean_probabilities = all_class_probabilities

    # Create a bar plot
    # mean_probabilities
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(colors.keys(), mean_probabilities, color=[colors[emotion] for emotion in colors])
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Mean Probability (%)')
    ax.set_title('Mean Emotion Probabilities')
    ax.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)




def main():
    st.title("Emotion Detection App")

    # Button to start camera and record
    if st.button("Start Camera"):
        av_recorder.start_rec()
        
        ## Separate audio and video
        #seperator(input_file)
        
        # Call video_app function to get combined_images and confusion_matrix_image
        combined_images, _, vid_fps, confusion_matrix_image = vp.video_app()
        skips=2
        
        # Display emotion probability plot and performance metrics
        all_class_probabilities = np.random.rand(7) * 100  # Random probabilities for each emotion
        all_class_probabilities1 = np.random.rand(7) * 100
        probab_class_neg = (all_class_probabilities[0] + all_class_probabilities[1] + all_class_probabilities[2] + all_class_probabilities[3])/4
        probab_class_neg
        probab_class_pos = (all_class_probabilities[3] + all_class_probabilities[6])/2
        probab_class_pos
        all_class_probabilities[4]
        # all_class_probabilities[0]
        emotion_probability_plot(all_class_probabilities)
        # Display the confusion matrix
        st.image(confusion_matrix_image, caption='Confusion Matrix')


if __name__ == "__main__":
    main()
