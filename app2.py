from fer import FER
from fer import Video
import pandas as pd
import os
import numpy as np
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
import audi_app
## Import modules
import av_recorder
import plotly.express as px


current_directory = os.getcwd()
video_file = current_directory+ "/"+'videos/output.avi'
audio_file = current_directory+ "/"+"audios/temp_audio.wav"


## Video processing
def record_video(video_file):
    face_detector = FER(mtcnn=True)
    preprocess_video = Video(video_file=video_file)
    preprocessing_vid = preprocess_video.analyze(face_detector,display=True)
    
    preprocess_df = pd.DataFrame(preprocessing_vid)
    df_mean = preprocess_df[["angry0",'disgust0','fear0','happy0','neutral0','sad0','surprise0']].mean()
    df_mean = df_mean*100
    df_mean = df_mean[["angry0",'disgust0','fear0','happy0','neutral0','sad0','surprise0']]
    df_mean['neg'] = df_mean[['angry0', 'disgust0', 'fear0', 'sad0']].sum()
    df_mean['neu'] = df_mean['neutral0']
    df_mean['pos'] = df_mean[['happy0', 'surprise0']].sum()
    df_mean['compound'] = df_mean[['neg', 'pos', 'neu']].sum()
    # df_mean.drop(columns=['angry0', 'disgust0', 'fear0', 'happy0', 'neutral0', 'sad0', 'surprise0'], inplace=True)
    df_mean = df_mean[['neg', 'neu', 'pos', 'compound']]
    
    
    # Melt the DataFrame for Plotly
    # df_mean = df_mean.melt(id_vars='index', var_name='Sentiment', value_name='Score')

    return df_mean
    # # # Create a bar chart
    # fig = px.bar(df_mean, x=df_mean.index, y=df_mean.values, text=df_mean.values)

    # # Update the layout for better visualization
    # fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    # # Customize hover info
    # fig.update_traces(hoverinfo='text+name', hovertext=df_mean.values)
    
    # fig.update_xaxes(title_text="Sentiment")
    # fig.update_yaxes(title_text="Score")

    # # Show the figure
    # st.plotly_chart(fig)

    
## Audio processing
def process_audio(audio_file):
    df_audio = audi_app.au_record(audio_file)

    # fig = px.bar(df_audio, x='Sentiment', y='Score', labels={'x': 'Sentiment', 'y': 'Score'}, title='Sentiment Analysis')
    # st.plotly_chart(fig)
    
    return df_audio

## Merging audio and video
def au_and_video(video_file,audio_file):
    video_df = record_video(video_file)
    audio_df = process_audio(audio_file)
    
    video_df1 = pd.DataFrame(video_df)
    video_df1.reset_index(inplace=True)

    # Reshape the DataFrame
    video_df1 = video_df1.melt(id_vars='index', var_name='Sentiment', value_name='Score')

    # Rename the index column to 'Sentiment'
    video_df1 = video_df1.drop(columns='Sentiment',axis=1)
    video_df1.rename(columns={'index': 'Sentiment'}, inplace=True)

    
    vid_score = video_df1['Score']

    audio_score = (audio_df['Score'])*100

    df_merge = (vid_score+audio_score)/2
    # df_merge = (video_df+audio_df)/2
    row_names = ['neg', 'neu', 'pos', 'compound']
    df_merge.index = row_names    
    # s= df_merge.values
    # s
    
    return video_df , audio_df , df_merge
            
def main():
    st.title("Emotion Detection App")

    # Button to start camera and record
    if st.button("Start Camera"):
        av_recorder.start_rec()
        
        # record_video(video_file)
        # process_audio(audio_file)
    video_df , audio_df , df_merge =  au_and_video(video_file,audio_file)
        
    option = st.selectbox(
        "Select on the basis",
        ("Video", "Audio", "Video&Audio")
    )
    
    if option == 'Video':
            ### Video 
            fig = px.bar(video_df, x=video_df.index, y=video_df.values, text=video_df.values)

            # Update the layout for better visualization
            fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

            # Customize hover info
            fig.update_traces(hoverinfo='text+name', hovertext=video_df.values)
            
            fig.update_xaxes(title_text="Sentiment")
            fig.update_yaxes(title_text="Score")
            
                # Show the figure
            st.plotly_chart(fig)
            
    elif option == 'Audio':
        ### Audio
        fig = px.bar(audio_df, x='Sentiment', y='Score', labels={'x': 'Sentiment', 'y': 'Score'})
        st.plotly_chart(fig)
        
    ### Merged
    elif option == "Video&Audio":
        # Create the plotly figure
        fig = px.bar(df_merge, x=df_merge.index, y='Score')
        st.plotly_chart(fig)
    
    else:
                    ### Video 
        fig = px.bar(video_df, x=video_df.index, y=video_df.values, text=video_df.values)

        # Update the layout for better visualization
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

        # Customize hover info
        fig.update_traces(hoverinfo='text+name', hovertext=video_df.values)
        
        fig.update_xaxes(title_text="Sentiment")
        fig.update_yaxes(title_text="Score")
        
            # Show the figure
        st.plotly_chart(fig)
        
    
if __name__ == "__main__":
    main()