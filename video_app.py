import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from moviepy.editor import ImageSequenceClip
from tqdm.notebook import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import torch
from facenet_pytorch import MTCNN
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoConfig
import os
import cv2
import seaborn as sns
import av_recorder
from av_seprator import seperator
#from audio_app import inference
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import (AutoFeatureExtractor,
                          AutoModelForImageClassification,
                          AutoConfig)

from PIL import Image, ImageDraw

# Set cache directories for XDG and Hugging Face Hub
os.environ['XDG_CACHE_HOME'] = '/home/msds2023/jlegara/.cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/msds2023/jlegara/.cache'


def video_app():
    # Load video and get frames per second
    video_data, vid_fps = video_capture("videos/output_with_audio.mp4")
    
    # Process video frames
    combined_images = process_video(video_data)
    
    
    # Generate confusion matrix image
    confusion_matrix_image = generate_confusion_matrix_image()
   
    predicted_labels = np.random.choice(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], len(combined_images))
    ground_truth_labels = np.random.choice(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], len(predicted_labels))
   
    accuracy = compute_accuracy(ground_truth_labels, predicted_labels)
    #st.write(f"Accuracy: {accuracy}")

    
    return combined_images, vid_fps, None, confusion_matrix_image  # Replace None with the appropriate value


# scene = 'videos/output1.avi'
def video_capture(scene):
    # Open the video file
    video_capture = cv2.VideoCapture(scene)

    # Get the frames per second of the video
    vid_fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Initialize an empty list to store frames
    video_data = []

    # Read frames from the video
    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()

        # If the frame was not successfully read, break the loop
        if not ret:
            break

        # Convert the frame from BGR to RGB (OpenCV uses BGR, but MoviePy uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Append the frame to the list
        video_data.append(frame_rgb)

    # Release the video capture object
    video_capture.release()

    # Convert the list of frames to a numpy array
    video_data = np.array(video_data)
    # print(video_data.shape)
    
    return video_data, vid_fps


def process_video(video_data,max_frames= 100):
    # Placeholder for combined images
    combined_images = []

    # Loop over video frames
    for frame in tqdm(video_data[:max_frames], desc="Processing frames"):
    # Convert frame to PIL image
        pil_image = Image.fromarray(frame)

        # Do something with the frame (placeholder example)
        # In real scenario, you'd apply your model here
        # For demonstration, just adding frame to combined_images
        combined_images.append(np.array(pil_image))

    return combined_images


def detect_emotions(image):
    # Initialize MTCNN model for single face cropping
    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=200,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        keep_all=False
    )

    # Load the pre-trained model and feature extractor
    extractor = AutoFeatureExtractor.from_pretrained(
        "trpakov/vit-face-expression"
    )
    model = AutoModelForImageClassification.from_pretrained(
        "trpakov/vit-face-expression"
    )
    


    """
    Detect emotions from a given image.
    Returns a tuple of the cropped face image and a
    dictionary of class probabilities.
    """
    
    temporary = image.copy()

    # Detect faces in the image using the MTCNN group model
    sample = mtcnn.detect(temporary)
    if sample[0] is not None:
        box = sample[0][0]

        # Crop the face
        face = temporary.crop(box)

        # Pre-process the face
        inputs = extractor(images=face, return_tensors="pt")

        # Run the image through the model
        outputs = model(**inputs)

        # Apply softmax to the logits to get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits,
                                                    dim=-1)

        # Retrieve the id2label attribute from the configuration
        config = AutoConfig.from_pretrained(
            "trpakov/vit-face-expression"
        )
        id2label = config.id2label

        # Convert probabilities tensor to a Python list
        probabilities = probabilities.detach().numpy().tolist()[0]

        # Map class labels to their probabilities
        class_probabilities = {
            id2label[i]: prob for i, prob in enumerate(probabilities)
        }
        
        # print(face,class_probabilities)
        return face, class_probabilities
    return None, None

def create_combined_image(face, class_probabilities):
    """
    Create an image combining the detected face and a barplot
    of the emotion probabilities.

    Parameters:
    face (PIL.Image): The detected face.
    class_probabilities (dict): The probabilities of each
        emotion class.

    Returns:
    np.array: The combined image as a numpy array.
    """
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
    palette = [colors[label] for label in class_probabilities.keys()]

    # Create a figure with 2 subplots: one for the
    # face image, one for the barplot
    fig, axs = plt.subplots(figsize=(8, 3))

    # Display face on the left subplot
    # axs[0].imshow(np.array(face))
    # axs[0].axis('off')

    # Create a barplot of the emotion probabilities
    # on the right subplot
    sns.barplot(ax=axs,
                y=list(class_probabilities.keys()),
                x=[prob * 100 for prob in class_probabilities.values()],
                palette=palette,
                orient='h')
    axs.set_xlabel('Probability (%)')
    axs.set_title('Emotion Probabilities')
    axs.set_xlim([0, 10])  # Set x-axis limits

    # Convert the figure to a numpy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    width_height = canvas.get_width_height()
    print("Width and height:", width_height)
    img = img.reshape(width_height[::-1] + (3,))




    plt.close(fig)
    # print(img)
    return img





def reduced_videos(scene):
    video_data,_ = video_capture(scene)
    skips = 2
    reduced_video = []

    for i in tqdm(range(0, len(video_data), skips)):
        reduced_video.append(video_data[i])
        
    return reduced_video

def proba(scene, max_frames=100):
    reduced_videos = reduced_videos(scene)[:max_frames]  # Limit the number of frames

    # Define a list of emotions
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    # List to hold the combined images
    combined_images = []

    # Create a list to hold the class probabilities for all frames
    all_class_probabilities = []

    # Loop over video frames
    for i, frame in tqdm(enumerate(reduced_videos), total=min(len(reduced_videos), max_frames), desc="Processing frames"):
        # Convert frame to uint8
        frame = frame.astype(np.uint8)

        # Call detect_emotions to get face and class probabilities
        face, class_probabilities = detect_emotions(Image.fromarray(frame))

        # If a face was found
        if face is not None:
            # Create combined image for this frame
            combined_image = create_combined_image(face, class_probabilities)

            # Append combined image to the list
            combined_images.append(combined_image)
        else:
            # If no face was found, set class probabilities to None
            class_probabilities = {emotion: None for emotion in emotions}

        # Append class probabilities to the list
        all_class_probabilities.append(class_probabilities)

    # Save combined images to temporary files
    temp_image_files = []
    for i, img in enumerate(combined_images):
        temp_image_file = f"temp_image_{i}.png"
        Image.fromarray(img).save(temp_image_file)
        temp_image_files.append(temp_image_file)
        
    predicted_labels = []
    for probs in all_class_probabilities:
        if probs is not None and any(probs.values()):  # Check if probs is not None and contains non-zero probabilities
            predicted_label = max(probs, key=probs.get)
        else:
            predicted_label = None
        predicted_labels.append(predicted_label)

    # Generate confusion matrix
    confusion_matrix_image = generate_confusion_matrix_image()
    # Generate random predicted labels
    predicted_labels = np.random.choice(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], len(combined_images))
    
    return combined_images, temp_image_files, predicted_labels, confusion_matrix_image


def create_video_clip(combined_images, vid_fps, skips):
    print("Length of combined_images:", len(combined_images))
    print("Combined images:", combined_images)
    clip_with_plot = ImageSequenceClip(combined_images, fps=vid_fps/skips)
    return clip_with_plot

# def compute_performance_matrix(ground_truth_labels, predicted_labels, emotions):
#     performance_matrix = {}

#     for emotion in emotions:
#         true_indices = [i for i, label in enumerate(ground_truth_labels) if label == emotion]
        
#         if predicted_labels is not None:
#             pred_indices = [i for i, label in enumerate(predicted_labels) if label == emotion]

#             accuracy = accuracy_score([ground_truth_labels[i] for i in true_indices],
#                                       [predicted_labels[i] for i in true_indices])
#             precision = precision_score([ground_truth_labels[i] for i in true_indices],
#                                         [predicted_labels[i] for i in true_indices])
#             recall = recall_score([ground_truth_labels[i] for i in true_indices],
#                                   [predicted_labels[i] for i in true_indices])
#             f1 = f1_score([ground_truth_labels[i] for i in true_indices],
#                           [predicted_labels[i] for i in true_indices])

#         performance_matrix[emotion] = {
#             'Accuracy': accuracy,
#             'Precision': precision,
#             'Recall': recall,
#             'F1-score': f1
#         }

#     return performance_matrix


def generate_confusion_matrix_image():
    # Generate random confusion matrix (replace with your actual confusion matrix)
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    confusion_matrix_data = np.random.rand(len(classes), len(classes))
    
    # Normalize confusion matrix
    confusion_matrix_data = confusion_matrix_data / confusion_matrix_data.sum(axis=1, keepdims=True)
    
    # Create heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    sns.heatmap(confusion_matrix_data, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes, fmt=".2f")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # Save confusion matrix as image
    confusion_matrix_image_path = 'confusion_matrix.png'
    plt.savefig(confusion_matrix_image_path)
    
    # Close plot to prevent displaying it in streamlit
    plt.close()
    
    # Load saved image and return as numpy array
    return np.array(Image.open(confusion_matrix_image_path))

def compute_accuracy(ground_truth_labels, predicted_labels):
    return np.mean(ground_truth_labels == predicted_labels)