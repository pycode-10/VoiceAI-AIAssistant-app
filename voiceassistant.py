import os
import time
from groq import Groq
import google.generativeai as genai
from PIL import ImageGrab, Image
import pyperclip
from faster_whisper import WhisperModel
import pyttsx3
import speech_recognition as sr
import re
from dotenv import load_dotenv

load_dotenv()

wake_word = 'alexa'

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


groq_model = Groq(api_key=os.getenv("groq_api_key"))
google_api_key = os.getenv("google_api_key")
genai.configure(api_key=google_api_key)

num_cores = os.cpu_count()
whisper_size = 'base'
whisper_model = WhisperModel(whisper_size, device='cpu', compute_type='int8', cpu_threads=num_cores//2, num_workers=num_cores//2)

r = sr.Recognizer()
source = sr.Microphone()

generation_config = {
    "temperature": 0.8,
    "max_output_tokens": 2048,
    "top_k": 1,
    "top_p": 1,
    "stop_sequences": ["\n"],
}

safety_settings = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE'
    },
]

system_msg_llama = (
    '''You are an AI voice assistant. Your task is to generate accurate response to the user prompt. The user may ask a direct question or may ask a question based on a given image. If the user has provided an image, it has already been processed and a detailed text prompt has been generated from it. The user image is not necessary.Generate accurate and factual response by considering all previous messages. Make sure that your response is clear and without any ambiguity. Just stay to the point and do not give extra explanations. Give proper and accurate response without any random additions. ALso if no information is present in the conversation, apply general knowledge. If you do not understand the question, also apply general knowledge to answer the question. Do not expect or request images, just use the context if added. Use all of the context of this conversation so your response is relevant to the conversation. Make your responses clear and concise, avoiding any verbosity.'''
)

conversation = [{'role': 'system', 'content': system_msg_llama}]

model = genai.GenerativeModel(model_name='gemini-1.5-flash', generation_config=generation_config, safety_settings=safety_settings)


def groq_function(prompt, img_context):
    if img_context:
        prompt = f"USER PROMPT: {prompt} \n IMAGE CONTEXT: {img_context}"
    conversation.append({'role': 'user', 'content': prompt})
    
    chat_completion = groq_model.chat.completions.create(messages=conversation, model='llama-3.1-8b-instant')
    response = chat_completion.choices[0].message
    
    conversation.append(response)

    return response.content

def function_call(prompt):
    system_message = (
        '''You are an AI model responsible for selecting the best function to execute in response to a user prompt. The available actions you can choose from are: 'take screenshot', 'extract clipboard', or 'no function'. Consider that the webcam is a standard laptop camera facing the user. Your reply must be exactly one of these options, without any extra text or reasoning. Choose the option that most appropriately fits the user’s request. Just stay to the point and do not give extra explanations. Give proper and accurate response without any random additions.'''
    )
    function_convo = [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': prompt}]
    chat_completion = groq_model.chat.completions.create(messages=function_convo, model='llama-3.1-8b-instant')
    response = chat_completion.choices[0].message
    return response.content

def take_screenshot():
    path = "screenshot.jpg"
    screenshot = ImageGrab.grab()
    screenshot.save(path)
    return path

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print("No text found in clipboard.")
        return None
    
def image_processing_model(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        '''You are an AI specialized in analyzing images to interpret and extract semantic information that adds context for another AI responsible for crafting a user response. Do not reply directly to the user as an assistant. Instead, use the user’s prompt along with the image to thoroughly identify and describe all relevant details visible in the image. Provide a detailed and exhaustive explanation of the image content related to the prompt. Afterwards, produce structured information that will be passed to the subsequent AI for generating the final user reply.
        USER PROMPT: {prompt}'''
    )
    response = model.generate_content([prompt, img])
    return response.text

def speak_response(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 190) 
    engine.setProperty('volume', 1.0)  
    voices = engine.getProperty('voices')
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.say(text)
    engine.runAndWait()
    


def wav_to_text(audio_path):
    result = whisper_model.transcribe(audio_path)
    segments = result[0]  
    text = ""
    for segment in segments:
        text += segment.text
    return text


def start_listening():
    with source as audio_source:
        r.adjust_for_ambient_noise(audio_source, duration=2)

    print("\nSay 'alexa' followed by your prompt...\n")

    stop_listener = r.listen_in_background(source, main_function)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_listener(wait_for_stop=False)


def extract_prompt(transcribed_text, wake_word):
    escaped_word = re.escape(wake_word)
    regex = rf'\b{escaped_word}[\s,.?!]*([^\s].*)'

    found = re.search(regex, transcribed_text, flags=re.IGNORECASE)

    if found:
        return found.group(1).strip()
    return None

def main_function(recognizer, audio):
    audio_path = "user_prompt.wav"

    with open(audio_path, "wb") as file:
        file.write(audio.get_wav_data())

    transcribed_text = wav_to_text(audio_path)
    user_prompt = extract_prompt(transcribed_text, wake_word)

    if user_prompt:
        print(f"USER: {user_prompt}")
        action = function_call(user_prompt)

        visual_context = None  

        if 'take screenshot' in action:
            print("Taking screenshot...")
            take_screenshot()
            visual_context = image_processing_model(prompt=user_prompt, photo_path='screenshot.jpg')
        
        elif 'extract clipboard' in action:
            print("Extracting clipboard text...")
            clipboard_content = get_clipboard_text()
            user_prompt += f"\nCLIPBOARD CONTENT: {clipboard_content}"

        ai_response = groq_function(prompt=user_prompt, img_context=visual_context)
        print("AI:", ai_response)
        speak_response(ai_response)

start_listening()
