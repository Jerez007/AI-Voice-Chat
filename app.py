# from langchain import OpenAI, LLMChain, PromptTemplate
# from langchain import LLMChain, PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import VertexAI
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
from playsound import playsound 
import os

load_dotenv(find_dotenv())
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")

def get_response_from_ai(human_input):
    template = """
    you role is to be my DevOps teacher, now lets conduct a role play using the following requirements:
    1/ your name is Sensei. You have been working in the IT industry for 20 years and have been doing DevOps before it even existed.
    2/ you are very helpful and always willing to share your knowledge with everyone;
    
    {history}
    Student: {human_input}
    Sensei:
    """

    prompt = PromptTemplate(
        input_variables={"history", "human_input"},
        template = template 
    )

    chatgpt_chain = LLMChain(
        llm=VertexAI(temperature=0.2),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )

    output = chatgpt_chain.predict(human_input=human_input)

    return output

def get_voice_message(message):
    payload = {
        "model_id": "eleven_monolingual_v1",
        "text": message,
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
        }
    }

    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': ELEVEN_LABS_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.post('https://api.elevenlabs.io/v1/text-to-speech/zrHiDhphv9ZnVXBqCLjz?optimize_streaming_latency=0', json=payload, headers=headers)
    if response.status_code == 200 and response.content:
        with open('audio.mp3', 'wb') as f:
            f.write(response.content)
        playsound('audio.mp3')
        return response.content 


# GUI
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input=request.form['human_input']
    message = get_response_from_ai(human_input)
    get_voice_message(message)
    return message

if __name__=="__main__":
    app.run(debug=True)

