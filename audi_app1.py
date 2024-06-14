from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr
import os

# Path to the audio file
current_directory = os.getcwd()
wav_file_path = current_directory+ "/"+"audios/temp_audio.wav"

def audio_to_text(wav_file_path):
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(wav_file_path) as source:
        print('Cleaning background noise....')
        audio_data = recognizer.record(source)

    # Recognize (convert from speech to text)
    try:
        text = recognizer.recognize_google(audio_data)
        # Sentence = [str(text)]

        # analyser = SentimentIntensityAnalyzer()
        # for i in Sentence:
        #     v =analyser.polarity_scores(i)
        # return v
        
    except Exception as ex:
        print(ex)
        # return "Google Speech Recognition could not understand the audio"
    # except sr.RequestError as e:
    #     return f"Could not request results from Google Speech Recognition service; {e}"
    
    
    ## Sentiment analysis
    Sentence = [str(text)]

    analyser = SentimentIntensityAnalyzer()
    for i in Sentence:
        v =analyser.polarity_scores(i)
    return v

# a = audio_to_text(wav_file_path)
# print(a)