# 🔊 VoiceAI: Your AI Voice Assistant

## 🧠 Introduction

**VoiceAI** is a multimodal AI-powered voice assistant that listens to your commands, understands visual and textual context, and responds intelligently using state-of-the-art models like **Groq's LLaMA 3** and **Google Gemini**. It combines speech recognition, image processing, clipboard reading, and conversational AI to deliver a seamless hands-free experience.

---

## ✨ Features

- 🗣️ **Wake Word Activation** — Activate with a simple "alexa"
- 🧠 **Handles Day-to-Day Conversations** — Ask anything, from general questions to contextual queries
- 🖼️ **Screen Understanding** — Takes screenshots and intelligently describes what’s on your screen
- 📋 **Clipboard Awareness** — Automatically reads and understands clipboard text
- 🎤 **Voice-to-Text** — Uses `faster-whisper` for efficient transcription
- 🧾 **Text-to-Speech** — Responds using natural speech output
- 💬 **Contextual AI Responses** — Powered by Groq LLaMA 3.1 and Gemini 1.5 Flash
- ⚙️ **Runs in Background** — Continually listens and responds without needing manual input

---

## 🚀 Installation

Follow these steps to set up and run the assistant:

### 1. Clone the Repository

```bash
git clone https://github.com/pycode-10/VoiceAI-AIAssistant-app.git
cd VoiceAI-AIAssistant-app
```
### 2. Create a .env File
Inside the project root, create a file named .env and add your API keys:
```bash
groq_api_key=your_groq_api_key_here
google_api_key=your_gemini_api_key_here
```
### 3. Install Requirements
```bash
pip install -r requirements.txt
```
### 4. Run the Voice Assistant
```bash
python voiceassistant.py
```
### 📌 Notes
Ensure your microphone and screen capture permissions are granted.

Works smoothly on CPU thanks to optimized faster-whisper.

Tested on Windows; may require slight modifications for Linux/macOS.
