import tkinter as tk
from tkinter import scrolledtext
import threading
import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import pyjokes
import code1  # Assuming code1 is the module where your chatbot model is defined

# Function to get response from chatbot
def get_response(question):
    response = code1.Pipe.predict([question])[0]
    return response

# Initialize speech recognition
recognizer = sr.Recognizer()

# Initialize text to speech 
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speech_to_text():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand what you said."
    except sr.RequestError as e:
        return "Could not request results from Google Speech Recognition service; {0}".format(e)

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

def searcher(command):
    if 'play' in command:
        song = command.replace('play', '')
        text_to_speech('playing ' + song)
        pywhatkit.playonyt(song)
    elif 'time' in command:
        time = datetime.datetime.now().strftime('%I:%M %p')
        text_to_speech('Current time is ' + time)
    elif 'who is' in command:
        person = command.replace('who is', '')
        info = wikipedia.summary(person, 1)
        text_to_speech(info)
    elif 'joke' in command:
        text_to_speech(pyjokes.get_joke())
    elif 'stop listening' in command:
        text_to_speech("okay")
        stop_listening()
    else:
        text_to_speech('Please say the command again.')

def handle_speech():
    user_input = speech_to_text()
    if user_input not in ["None", ""]:
        conversation_text.insert(tk.END, "You: " + user_input + "\n")
        if any(command in user_input.lower() for command in ['play', 'time', 'who is', 'joke','stop listening','Sorry']):
            searcher(user_input.lower())
        else:
            response = get_response(user_input)
            conversation_text.insert(tk.END, "Bot: " + response + "\n")
            text_to_speech(response)

def start_listening():
    listening_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    listening_thread = threading.Thread(target=listen_continuous)
    listening_thread.start()

def stop_listening():
    global listening
    listening = False
    listening_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)

def listen_continuous():
    global listening
    listening = True
    while listening:
        handle_speech()

# Create the main window
app = tk.Tk()
app.title("Speech-to-Speech Chatbot")

# Create a scrolled text box to display the conversation
conversation_text = scrolledtext.ScrolledText(app, wrap=tk.WORD, width=50, height=20, font=("Arial", 12))
conversation_text.pack(padx=10, pady=10)

# Create a button to start listening
listening_button = tk.Button(app, text="Start Listening", command=start_listening, font=("Arial", 14))
listening_button.pack(padx=10, pady=10)

# Create a button to stop listening
stop_button = tk.Button(app, text="Stop Listening", command=stop_listening, font=("Arial", 14), state=tk.DISABLED)
stop_button.pack(padx=10, pady=10)

# Run the app
app.mainloop()
