# Chatterbox FastRTC Realtime Emotion (Local)

Real-time conversational AI with voice cloning and emotion detection. Analyses conversation context to deliver dramatically expressive responses using your cloned voice. Built with FastRTC and Chatterbox TTS for natural, emotionally-aware voice interactions.


## ✨ Features

- 🎭 **Voice Cloning**: Use any voice from a single reference audio file
- 🎯 **Natural Emotion Detection**: Analyses conversation context to detect emotions automatically
- 🎪 **Dramatic Expression**: Dynamic voice synthesis with exaggeration, temperature, and cfg_weight adjustments
- ⚡ **Real-time Streaming**: Low-latency audio generation and playback
- 💬 **Dual Interface**: WebSocket text chat and Gradio voice chat
- 🧠 **Smart Context**: Maintains conversation history with emotional awareness
- 🎵 **12 Set Emotions**: Excited, happy, sad, angry, surprised, confused, tired, worried, calm, frustrated, enthusiastic, neutral

[![Watch the video](https://img.youtube.com/vi/ucWV44D5rW0/hqdefault.jpg)](https://youtu.be/ucWV44D5rW0)

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (RTX 4090 recommended for real-time performance)
- Ollama with Gemma 3 4B model

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/dwain-barnes/chatterbox-fastrtc-realtime-emotion.git
   cd chatterbox-fastrtc-realtime-emotion
   ```

2. **Install PyTorch for your system**
   ```bash
   # For CUDA 11.8 (check pytorch.org for your specific setup)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Chatterbox TTS (avoiding numpy conflicts)**
   ```bash
   # Important: Install without dependencies to avoid numpy==1.26.0 conflicts
   pip install --no-deps chatterbox-tts
   ```

5. **Install and run Ollama with Gemma 3 4B**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull gemma3:4b:latest
   ollama serve
   ```

6. **Add your voice reference (optional)**
   ```bash
   # Place your reference voice file in the project directory
   cp /path/to/your/voice.wav reference_voice.wav
   ```

## 🎮 Usage

### Start the Application

```bash
python realtime_emotion.py
```

### Access the Interfaces

- **Text Chat**: http://localhost:8000/
- **Voice Chat**: http://localhost:8000/gradio

### Voice Cloning Setup

1. Record a 10-30 second clear audio sample of the target voice
2. Save it as `reference_voice.wav` in the project directory
3. Restart the application
4. The cloned voice will be used for all emotional responses

## ⚙️ Technical Details

### Emotion Parameters

Each emotion uses carefully tuned parameters for dramatic expression:

- **Exaggeration**: 0.05 (tired) to 0.95 (excited)
- **CFG Weight**: 0.2 (angry) to 0.95 (tired)  
- **Temperature**: 0.3 (tired) to 1.3 (excited)

### Performance Requirements

- **Recommended**: RTX 4090 GPU for real-time generation
- **Minimum**: RTX 3070 or equivalent
- **Model**: Gemma 3 4B for optimal speed/quality balance
- **RAM**: 16GB+ recommended

### Architecture

- **Frontend**: FastAPI + WebSocket + HTML/CSS/JS
- **Voice Interface**: Gradio + FastRTC
- **TTS**: Chatterbox TTS with voice cloning
- **STT**: FastRTC STT model
- **LLM**: Ollama (Gemma 3 4B)
- **Emotion Detection**: Context-based pattern matching

## 🎯 How It Works

1. **Input Processing**: Text or voice input is received
2. **LLM Response**: Gemma 3 generates contextual response
3. **Emotion Detection**: Analyses response text for emotional patterns
4. **Voice Synthesis**: Applies dramatic parameters based on detected emotion
5. **Real-time Streaming**: Audio chunks streamed as they're generated
6. **Playback**: Client receives and plays audio with minimal latency

## 🔧 Configuration

### Emotion Tuning

Modify `EMOTION_PARAMETERS` in the code to adjust emotional expression:

```python
"excited": {
    "exaggeration": 0.95,    # Higher = more expressive
    "cfg_weight": 0.2,       # Lower = more variation
    "temperature": 1.3       # Higher = more dynamic
}
```

### Model Settings

- Change LLM model in the `init_chat_model` call
- Adjust chunk duration for latency vs quality trade-offs
- Modify sample rates for different audio quality

## 📝 Requirements

Key dependencies include:
- `fastapi` - Web framework
- `fastrtc` - Real-time communication
- `chatterbox-tts` - Voice synthesis and cloning
- `langchain` - LLM integration
- `gradio` - Voice interface
- `torch` - Deep learning framework
- `numpy` - Numerical computing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with different emotions and voices
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [Chatterbox TTS Streaming](https://github.com/davidbrowne17/chatterbox-streaming) for TTS
- [FastRTC](https://github.com/gradio-app/fastrtc) for real-time communication
- [Ollama](https://ollama.ai) for local LLM serving


**Experience emotional conversations with your own cloned voice! 🎭🎤**
