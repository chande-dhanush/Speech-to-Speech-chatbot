import time
import speech_recognition as sr
import pyttsx3
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import pywhatkit
import datetime
import wikipedia
import pyjokes
import requests
import tkinter.font as tkFont  # Correct import for tkinter font

# Set your Hugging Face API key here
HUGGING_FACE_API_KEY = 'Enter your hugging face API key'

API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"

headers = {
    "Authorization": f"Bearer {HUGGING_FACE_API_KEY}"
}

# Initialize speech recognition and text to speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def handle_speech():
    user_input = speech_to_text()
    if user_input not in ["None", ""]:
        process_input(user_input)

def handle_text():
    print("Handling text input...")  # Debugging statement
    user_input = text_entry.get()
    print(f"User input: {user_input}")  # Debugging statement
    if user_input not in ["None", ""]:
        process_input(user_input)
        text_entry.delete(0, tk.END)

def process_input(user_input):
    print(f"Processing input: {user_input}")  # Debugging statement
    conversation_text.insert(tk.END, "You: " + user_input + "\n")
    if any(command in user_input.lower() for command in ['play', 'who is', 'joke', 'stop listening']):
        searcher(user_input.lower())
    else:
        response = generate_response(user_input)
        conversation_text.insert(tk.END, "Bot: " + response + "\n")
        text_to_speech(response)
        time.sleep(2)

def generate_response(prompt, retries=5):
    data = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.8,
            "top_k": 60,
            "top_p": 0.95,
            "repetition_penalty": 2.5,
            "no_repeat_ngram_size": 3
        }
    }

    for attempt in range(retries):
        response = requests.post(API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            response_json = response.json()
            print(f"API Response: {response_json}")
            if isinstance(response_json, list) and 'generated_text' in response_json[0]:
                generated_text = response_json[0]['generated_text']
                return generated_text.strip()
            else:
                return "Sorry, the response format is not as expected."
        elif response.status_code == 503:
            print(f"Error: 503 - Model is currently loading, retrying in 20 seconds (Attempt {attempt + 1}/{retries})")
            time.sleep(10)
        else:
            print(f"Error: {response.status_code}")
            print(f"Response Content: {response.content}")
            return "Sorry, I couldn't generate a response."

    return "Sorry, the model is currently unavailable. Please try again later."

def searcher(command):
    if 'play' in command:
        song = command.replace('play', '')
        text_to_speech('playing ' + song)
        pywhatkit.playonyt(song)
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

def speech_to_text():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand what you said."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
    except sr.WaitTimeoutError:
        return "Listening timed out while waiting for phrase to start"

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

# Function to start listening
def start_listening():
    listening_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    listening_thread = threading.Thread(target=listen_continuous)
    listening_thread.start()

# Function to stop listening
def stop_listening():
    global listening
    listening = False
    listening_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    time.sleep(2)

def listen_continuous():
    global listening
    listening = True
    while listening:
        handle_speech()

# Create the main window
app = tk.Tk()
app.title("Speech-to-Speech Chatbot")

# Set window size and make it not resizable
app.geometry("600x800")
app.resizable(True, True)

# Set a nice font
font = tkFont.Font(family="Helvetica", size=12)

# Create a frame for the conversation text box
frame = tk.Frame(app, bg='#f5f5f5')
frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

# Create a scrolled text box to display the conversation
conversation_text = scrolledtext.ScrolledText(
    frame, wrap=tk.WORD, width=50, height=20, font=font, bg='#ffffff', fg='#333333'
)
conversation_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a frame for text entry
entry_frame = tk.Frame(app, bg='#f5f5f5')
entry_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

# Create a text entry box for user input
text_entry = tk.Entry(entry_frame, font=font, bg='#ffffff', fg='#333333')
text_entry.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
text_entry.bind('<Return>', lambda event: handle_text())

# Create a button frame
button_frame = tk.Frame(app, bg='#f5f5f5')
button_frame.pack(pady=10)

# Create a button to start listening
listening_button = tk.Button(
    button_frame, text="Start Listening", command=start_listening, font=("Helvetica", 14), bg='#4CAF50', fg='#ffffff', activebackground='#45a049'
)
listening_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X)

# Create a button to stop listening
stop_button = tk.Button(
    button_frame, text="Stop Listening", command=stop_listening, font=("Helvetica", 14), bg='#f44336', fg='#ffffff', activebackground='#d32f2f', state=tk.DISABLED
)
stop_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X)

# Create a button to send text input
send_button = tk.Button(
    button_frame, text="Send", command=handle_text, font=("Helvetica", 14), bg='#2196F3', fg='#ffffff', activebackground='#1e88e5'
)
send_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X)

# Run the app
app.mainloop()
