from fer import FER
from fer import Video
import pandas as pd
import os
import numpy as np

current_directory = os.getcwd()
video_file = current_directory+ "/"+'videos/output.avi'
# video_file = ""

face_detector = FER(mtcnn=True)

preprocess_video = Video(video_file=video_file)

preprocessing_vid = preprocess_video.analyze(face_detector,display=True)

# mean_pre = np.mean(pre)
preprocess_video.to_csv(preprocessing_vid,"data.csv")