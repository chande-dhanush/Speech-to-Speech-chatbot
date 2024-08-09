
from flask import Flask, render_template, request, jsonify
import code1
app = Flask(__name__)

# Function to get response from chatbot
def get_response(question):
    response = code1.Pipe.predict([question])[0]
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_chat_response():
    data = request.json
    question = data.get('question')
    response = get_response(question)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)