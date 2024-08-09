import speech_recognition as sr
import pyttsx3
import time
import tkinter as tk
from tkinter import scrolledtext
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import threading
import torch
import pywhatkit
import datetime
import wikipedia
import pyjokes
# Initialize the GPT-2 model and tokenizer
model_name = "gpt2-medium"  # You can use "gpt2" for a smaller model
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Initialize speech recognition and text to speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
def handle_speech():
    user_input = speech_to_text()
    if user_input not in ["None", ""]:
        conversation_text.insert(tk.END, "You: " + user_input + "\n")
        if any(command in user_input.lower() for command in ['play', 'time', 'who is', 'joke','stop listening','Sorry']):
            searcher(user_input.lower())
        else:
            response = generate_response(user_input)
            conversation_text.insert(tk.END, "Bot: " + response + "\n")
            text_to_speech(response)
            time.sleep(2) 

def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)

    # Generate response with additional repetition handling parameters
    outputs = model.generate(
        inputs,
        max_length=50,  # Reduced max_length
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.8,
        top_k=60,
        top_p=0.95,
        repetition_penalty=2.5,  # Increased repetition penalty
        no_repeat_ngram_size=3
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove extraneous text (e.g., repeating phrases and story-like content)
    response_lines = response.split('. ')
    filtered_lines = [line for line in response_lines if 'you' in line or 'I' in line or 'we' in line]
    final_response = '. '.join(filtered_lines[:2])  # Limit to the first 2 relevant sentences
    
    # Avoid echoing the user's input
    if prompt.lower() in final_response.lower():
        final_response = final_response.replace(prompt, "").strip()
    
    return final_response.strip()
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

def speech_to_text():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)  # Set timeout and phrase time limit

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

'''def handle_speech():
    user_input = speech_to_text()
    if user_input not in ["None", ""]:
        conversation_text.insert(tk.END, "You: " + user_input + "\n")
        response = generate_response(user_input)

        conversation_text.insert(tk.END, "Bot: " + response + "\n")
        text_to_speech(response)
        time.sleep(2)  # Pause for 2 seconds after responding
'''
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
