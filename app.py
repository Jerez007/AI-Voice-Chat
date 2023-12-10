from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
from playsound import playsound 
import os

load_dotenv(find_dotenv())

def get_response_from_ai(human_input):
    template = """
    you role is to be my DevOps teacher, now lets conduct a role play using the following requirements:
    1/ your name is Sensei. You have been working in the IT industry for 20 years and have been doing DevOps before it even existed.
    2/ you are very helpful and always willing to share your knowledge with everyone
    {history}
    Student: {human_input}
    Sensei:
    """

    prompt = PromptTemplate(
        input_variables={"history", "human_input"},
        template = template 
    )

    chatgpt_chain = LLMChain(
        lim=OpenAI(temperature=0.2),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )

    output = chatgpt_chain.predict(human_input=human_input)

    return output

# GUI
from flask import Flask, render_template, request

