#import library
import speech_recognition as sr
import os
#Initiаlize  reсоgnizer  сlаss  (fоr  reсоgnizing  the  sрeeсh)
r = sr.Recognizer()
# Reading Audio file as source
#  listening  the  аudiо  file  аnd  stоre  in  аudiо_text  vаriаble
current_directory = os.getcwd()
# req_directory = current_directory+ "/"+'audios/temp_audio.wav'

import os
import speech_recognition as sr

def audio_to_text(wav_file_path):
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(wav_file_path) as source:
        audio_data = recognizer.record(source)

    # Recognize (convert from speech to text)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

# Path to the audio file
wav_file_path = current_directory+ "/"+"audios/temp_audio.wav"

# Convert audio to text
transcription = audio_to_text(wav_file_path)
print("Transcription: ", transcription)

