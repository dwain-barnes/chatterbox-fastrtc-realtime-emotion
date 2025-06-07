from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AIMessageChunk
import nltk
import numpy as np
import torch
from chatterbox.tts import ChatterboxTTS # This is your chatterbox-streaming model
from chatterbox.models.s3tokenizer import S3_SR # STT model's input sample rate
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    get_stt_model,
    get_twilio_turn_credentials,
)
import gradio as gr
from fastrtc import Stream, ReplyOnPause, get_stt_model, AdditionalOutputs
import asyncio
import functools # For functools.partial with asyncio.to_thread
import json
import base64
from typing import Dict, Set, Optional
import logging
import time
import os
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Voice cloning configuration
AUDIO_PROMPT_PATH = "reference_voice.wav"  # Path to your reference voice file

# ── DRAMATIC EMOTION PARAMETERS FOR SINGLE CLONED VOICE ──────────────────────
# Made more extreme for noticeable differences with cloned voice
EMOTION_PARAMETERS: Dict[str, Dict[str, float]] = {
    "excited": {
        "exaggeration": 0.95,    # Very high expressiveness 
        "cfg_weight": 0.2,       # Low adherence for wild variation
        "temperature": 1.3       # High variation for energetic speech
    },
    "happy": {
        "exaggeration": 0.8,     # High expressiveness
        "cfg_weight": 0.3,       # Moderate adherence 
        "temperature": 1.1       # Good variation for cheerful tone
    },
    "enthusiastic": {
        "exaggeration": 0.9,     # Very high expressiveness
        "cfg_weight": 0.25,      # Low adherence for energy
        "temperature": 1.2       # High variation
    },
    "sad": {
        "exaggeration": 0.1,     # Very low expressiveness for monotone
        "cfg_weight": 0.9,       # High adherence for controlled tone
        "temperature": 0.4       # Low variation for flat delivery
    },
    "angry": {
        "exaggeration": 0.85,    # High expressiveness for intensity
        "cfg_weight": 0.2,       # Low adherence for aggressive variation
        "temperature": 1.0       # High variation for anger
    },
    "frustrated": {
        "exaggeration": 0.7,     # Moderate-high expressiveness
        "cfg_weight": 0.3,       # Moderate adherence
        "temperature": 0.9       # Controlled variation
    },
    "calm": {
        "exaggeration": 0.2,     # Low expressiveness for peaceful tone
        "cfg_weight": 0.8,       # High adherence for steady voice
        "temperature": 0.5       # Low variation for tranquility
    },
    "neutral": {
        "exaggeration": 0.5,     # Balanced baseline
        "cfg_weight": 0.5,       # Balanced adherence
        "temperature": 0.7       # Moderate variation
    },
    "confused": {
        "exaggeration": 0.4,     # Low-moderate expressiveness
        "cfg_weight": 0.6,       # Moderate adherence with some uncertainty
        "temperature": 0.8       # Some variation for questioning tone
    },
    "surprised": {
        "exaggeration": 0.8,     # High expressiveness for shock
        "cfg_weight": 0.3,       # Lower adherence for sudden change
        "temperature": 1.0       # High variation for surprise
    },
    "tired": {
        "exaggeration": 0.05,    # Extremely low expressiveness
        "cfg_weight": 0.95,      # Very high adherence for monotone
        "temperature": 0.3       # Very low variation for exhaustion
    },
    "worried": {
        "exaggeration": 0.3,     # Low expressiveness for anxiety
        "cfg_weight": 0.7,       # High adherence with controlled worry
        "temperature": 0.6       # Lower variation for nervousness
    },
}

# Initialize models and clients
stt_model = get_stt_model()

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"

tts_model = ChatterboxTTS.from_pretrained(device=DEVICE)

# Verify audio prompt file exists
if not os.path.exists(AUDIO_PROMPT_PATH):
    logger.warning(f"Voice reference file not found: {AUDIO_PROMPT_PATH}")
    logger.warning("Voice cloning will be disabled - using default voice")
    AUDIO_PROMPT_PATH = None
else:
    logger.info(f"Voice cloning enabled with reference: {AUDIO_PROMPT_PATH}")

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

llm = init_chat_model("ollama:gemma3:4b-it-qat", base_url="http://localhost:11434")

# Natural conversational system prompt for voice agent
SYSTEM_PROMPT = """You are a friendly, emotionally expressive voice assistant having natural conversations with users. 

Respond as a real person would - with genuine emotional reactions that fit the context:
- Get excited about cool topics, achievements, or interesting discoveries
- Feel genuinely sad or sympathetic when hearing about problems or losses  
- Express confusion when things don't make sense
- Show enthusiasm for things you find fascinating
- React with appropriate surprise to unexpected information
- Feel calm and peaceful when discussing relaxing topics

Speak naturally and conversationally, as if you're talking to a friend. Keep responses concise (1-2 sentences) since this is spoken conversation. Let your emotions show through your word choices and tone, just like humans do.

You have an expressive voice that will naturally convey your emotions, so focus on being genuine rather than performative."""

app = FastAPI()

# ── EMOTION DETECTION ───────────────────────────────────────────────────────
def detect_emotion_from_text(text: str, debug: bool = False) -> str:
    """Detect emotion from text based on natural emotional expressions and context"""
    tl = text.lower()
    
    # Look for natural emotional expressions, not just keywords
    emotional_patterns = [
        # Excitement/Joy patterns
        ("excited", [
            "so exciting", "can't wait", "thrilled", "pumped", "stoked",
            "this is great", "love this", "fantastic", "awesome", "amazing",
            "incredible", "wonderful", "brilliant", "perfect"
        ]),
        
        # Happiness patterns  
        ("happy", [
            "makes me happy", "feel good", "great news", "delighted",
            "pleased", "glad", "joyful", "cheerful", "upbeat", "positive",
            "excellent", "superb", "marvelous"
        ]),
        
        # Sadness patterns
        ("sad", [
            "sorry to hear", "that's sad", "feel bad", "disappointed",
            "heartbreaking", "tragic", "unfortunate", "terrible", "awful",
            "depressing", "devastating", "mourning", "grieving"
        ]),
        
        # Anger/Frustration patterns
        ("angry", [
            "that's ridiculous", "outrageous", "frustrating", "annoying",
            "makes me mad", "can't believe", "infuriating", "absurd",
            "unacceptable", "disgusting", "appalling"
        ]),
        
        # Surprise patterns
        ("surprised", [
            "wow", "no way", "really?", "seriously?", "can't believe",
            "shocking", "unexpected", "surprising", "astonishing",
            "unbelievable", "mind-blowing", "incredible"
        ]),
        
        # Confusion patterns
        ("confused", [
            "don't understand", "confused", "what do you mean", "unclear",
            "not sure", "puzzled", "baffled", "perplexed", "lost",
            "doesn't make sense", "how does", "why would"
        ]),
        
        # Worry/Concern patterns
        ("worried", [
            "worried about", "concerned", "anxious", "nervous", "scared",
            "frightened", "troubling", "disturbing", "alarming",
            "stress", "panic", "fear"
        ]),
        
        # Tiredness patterns
        ("tired", [
            "exhausted", "tired", "worn out", "drained", "weary",
            "sleepy", "fatigue", "burned out", "overwhelmed"
        ]),
        
        # Calm/Peace patterns
        ("calm", [
            "peaceful", "relaxing", "tranquil", "serene", "soothing",
            "calm", "gentle", "quiet", "restful", "meditative"
        ])
    ]
    
    # Check for emotional patterns
    for emotion, patterns in emotional_patterns:
        for pattern in patterns:
            if pattern in tl:
                if debug:
                    logger.info(f"🎭 Detected emotion: {emotion} (pattern: '{pattern}')")
                return emotion
    
    # Context-based detection for questions and statements
    if "?" in text:
        if any(word in tl for word in ["how", "why", "what", "when", "where"]):
            if debug:
                logger.info("🎭 Detected emotion: confused (questioning context)")
            return "confused"
    
    if debug:
        logger.info("🎭 No specific emotion detected, using neutral")
    return "neutral"

def get_emotion_parameters(emotion: str) -> Dict[str, float]:
    """Get dramatic emotion parameters for single cloned voice"""
    if emotion not in EMOTION_PARAMETERS:
        emotion = "neutral"
    
    params = EMOTION_PARAMETERS[emotion]
    
    return {
        "exaggeration": params["exaggeration"],
        "cfg_weight": params["cfg_weight"], 
        "temperature": params["temperature"],
    }

def clean_text_for_tts(text: str) -> str:
    """Clean text for TTS by removing problematic characters"""
    # Remove emojis and excessive whitespace
    emoji_pattern = re.compile(r"[\U00010000-\U0010FFFF]", flags=re.UNICODE)
    cleaned = emoji_pattern.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.chat_histories: Dict[str, list] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        if client_id not in self.chat_histories:
            self.chat_histories[client_id] = []
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")

    def get_chat_history(self, client_id: str) -> list:
        return self.chat_histories.get(client_id, [])

    def update_chat_history(self, client_id: str, history: list):
        self.chat_histories[client_id] = history

manager = ConnectionManager()

def convert_tensor_audio_to_int16(audio_tensor: torch.Tensor) -> np.ndarray:
    if audio_tensor.ndim > 1: audio_tensor = audio_tensor.flatten()
    raw_tts_audio = audio_tensor.cpu().numpy()
    processed_tts_audio_1d: np.ndarray
    if raw_tts_audio.dtype == np.float32 or raw_tts_audio.dtype == np.float64:
        if np.max(np.abs(raw_tts_audio)) > 1.0:
             raw_tts_audio = np.clip(raw_tts_audio, -1.0, 1.0)
        processed_tts_audio_1d = (raw_tts_audio * 32767).astype(np.int16)
    elif raw_tts_audio.dtype == np.int32:
        if raw_tts_audio.max() <= 32767 and raw_tts_audio.min() >= -32768:
            processed_tts_audio_1d = raw_tts_audio.astype(np.int16)
        else:
            processed_tts_audio_1d = (raw_tts_audio // (2**16)).astype(np.int16)
    elif raw_tts_audio.dtype == np.int16:
        processed_tts_audio_1d = raw_tts_audio
    else:
        try: processed_tts_audio_1d = raw_tts_audio.astype(np.int16)
        except Exception: processed_tts_audio_1d = np.array([], dtype=np.int16)
    if processed_tts_audio_1d.ndim == 0 and processed_tts_audio_1d.size == 1:
         processed_tts_audio_1d = np.array([processed_tts_audio_1d.item()], dtype=np.int16)
    return processed_tts_audio_1d

# --- Synchronous Helper for LLM Streaming in a Thread ---
def _threaded_llm_streamer(messages: list, queue: asyncio.Queue, llm_instance):
    try:
        for llm_chunk in llm_instance.stream(messages):
            queue.put_nowait(llm_chunk)
    except Exception as e:
        logger.error(f"Threaded LLM: Error during stream: {e}")
    finally:
        queue.put_nowait(None)

def _threaded_tts_generator_with_natural_emotion(
    text: str, 
    queue: asyncio.Queue, 
    tts_model_instance: ChatterboxTTS, 
    target_chunk_duration: float = 0.15, 
    audio_prompt_path: str = None,
    emotion: str = "neutral"
):
    """TTS generator with dramatic emotion parameters and cloned voice"""
    try:
        accumulated_audio = []
        target_samples = int(target_chunk_duration * tts_model_instance.sr)
        
        # Clean text for TTS
        clean_text = clean_text_for_tts(text)
        if not clean_text:
            logger.warning("Text is empty after cleaning, skipping TTS generation")
            return
        
        # Get dramatic emotion parameters
        emotion_params = get_emotion_parameters(emotion)
        
        logger.info(f"🎭 Generating natural TTS with CLONED VOICE: emotion={emotion}")
        logger.info(f"📊 Emotional Parameters: exaggeration={emotion_params['exaggeration']:.2f}, "
                   f"cfg_weight={emotion_params['cfg_weight']:.2f}, "
                   f"temperature={emotion_params['temperature']:.2f}")
        
        # Generate stream with dramatic emotion parameters and cloned voice
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            logger.info(f"🎤 Using CLONED voice with {emotion.upper()} emotion")
            stream_generator = tts_model_instance.generate_stream(
                clean_text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=emotion_params['exaggeration'],
                cfg_weight=emotion_params['cfg_weight'],
                temperature=emotion_params['temperature']
            )
        else:
            logger.warning("🎤 Voice cloning disabled - using default voice with emotion")
            stream_generator = tts_model_instance.generate_stream(
                clean_text,
                exaggeration=emotion_params['exaggeration'],
                cfg_weight=emotion_params['cfg_weight'],
                temperature=emotion_params['temperature']
            )
        
        for audio_chunk_tensor, metrics in stream_generator:
            accumulated_audio.append(audio_chunk_tensor)
            
            # Calculate total accumulated samples
            total_samples = sum(chunk.numel() for chunk in accumulated_audio)
            
            # If we have enough samples for the target duration, send chunk
            if total_samples >= target_samples:
                # Concatenate accumulated audio
                combined_audio = torch.cat(accumulated_audio, dim=-1)
                queue.put_nowait(combined_audio)
                accumulated_audio = []
                
        # Send any remaining audio
        if accumulated_audio:
            combined_audio = torch.cat(accumulated_audio, dim=-1)
            queue.put_nowait(combined_audio)
            
    except Exception as e:
        logger.error(f"Threaded TTS: Error during generation: {e}")
    finally:
        queue.put_nowait(None)

async def websocket_llm_tts_streaming(user_content: str, client_id: str, websocket: WebSocket):
    """WebSocket-optimized streaming function with dramatic emotion and single cloned voice"""
    current_chat_history = manager.get_chat_history(client_id)
    current_chat_history.append({"role": "user", "content": user_content})
    
    # Send chat update
    await manager.send_personal_message({
        "type": "chat_update",
        "data": list(current_chat_history)
    }, client_id)

    llm_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg_obj in list(current_chat_history):
        if msg_obj["role"] == "user": 
            llm_messages.append(HumanMessage(content=msg_obj["content"]))
        elif msg_obj["role"] == "assistant": 
            llm_messages.append(AIMessage(content=msg_obj["content"]))
    
    full_llm_response_text = ""
    assistant_message_index = len(current_chat_history)
    current_chat_history.append({"role": "assistant", "content": "▍"})
    
    # Send initial assistant message
    await manager.send_personal_message({
        "type": "chat_update",
        "data": list(current_chat_history)
    }, client_id)

    # Stream LLM response
    llm_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    llm_thread_task = loop.create_task(
        asyncio.to_thread(_threaded_llm_streamer, llm_messages, llm_queue, llm)
    )

    while True:
        llm_chunk = await llm_queue.get()
        if llm_chunk is None:
            llm_queue.task_done()
            break
        chunk_content = getattr(llm_chunk, 'content', '')
        if chunk_content:
            full_llm_response_text += chunk_content
            current_chat_history[assistant_message_index]["content"] = full_llm_response_text + "▍"
            await manager.send_personal_message({
                "type": "chat_update",
                "data": list(current_chat_history)
            }, client_id)
        llm_queue.task_done()
    
    await llm_thread_task

    # Update final response
    current_chat_history[assistant_message_index]["content"] = full_llm_response_text
    await manager.send_personal_message({
        "type": "chat_update",
        "data": list(current_chat_history)
    }, client_id)

    # Update chat history in manager
    manager.update_chat_history(client_id, current_chat_history)

    # Stream TTS audio with dramatic emotion and single cloned voice
    if full_llm_response_text.strip():
        # Detect emotion from the response text
        detected_emotion = detect_emotion_from_text(full_llm_response_text, debug=True)
        
        # Always use cloned voice if available, regardless of emotion
        voice_status = "cloned" if AUDIO_PROMPT_PATH else "default"
        
        # Send audio start with emotion and voice status
        await manager.send_personal_message({
            "type": "audio_start",
            "sample_rate": tts_model.sr,
            "voice_type": voice_status,
            "emotion": detected_emotion,
            "reference_file": AUDIO_PROMPT_PATH if AUDIO_PROMPT_PATH else None
        }, client_id)
        
        tts_queue = asyncio.Queue()
        tts_thread_task = loop.create_task(
            asyncio.to_thread(
                _threaded_tts_generator_with_natural_emotion, 
                full_llm_response_text, 
                tts_queue, 
                tts_model, 
                0.15,  # chunk duration
                AUDIO_PROMPT_PATH,  # voice cloning reference
                detected_emotion    # detected emotion with dramatic parameters
            )
        )
        
        chunk_count = 0
        start_time = time.time()
        
        while True:
            audio_chunk_tensor = await tts_queue.get()
            if audio_chunk_tensor is None:
                tts_queue.task_done()
                break
            
            processed_audio_1d = convert_tensor_audio_to_int16(audio_chunk_tensor)
            if processed_audio_1d.size > 0:
                audio_base64 = base64.b64encode(processed_audio_1d.tobytes()).decode('utf-8')
                
                # Add timing information for better client-side handling
                chunk_duration = len(processed_audio_1d) / tts_model.sr
                
                await manager.send_personal_message({
                    "type": "audio_chunk",
                    "sample_rate": tts_model.sr,
                    "data": audio_base64,
                    "chunk_size": len(processed_audio_1d),
                    "chunk_duration": chunk_duration,
                    "chunk_index": chunk_count,
                    "timestamp": time.time() - start_time,
                    "voice_type": voice_status,
                    "emotion": detected_emotion
                }, client_id)
                
                chunk_count += 1
                
                # Small delay to prevent overwhelming the WebSocket and ensure proper sequencing
                await asyncio.sleep(0.02)  # 20ms delay between chunks
                
            tts_queue.task_done()
        
        await tts_thread_task
        
        await manager.send_personal_message({
            "type": "audio_complete",
            "total_chunks": chunk_count,
            "total_duration": time.time() - start_time,
            "voice_type": voice_status,
            "emotion": detected_emotion
        }, client_id)
    else:
        logger.info("LLM response was empty, no TTS to stream.")

# WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "text_input":
                text_content = message["content"]
                if text_content and text_content.strip():
                    await websocket_llm_tts_streaming(text_content, client_id, websocket)
                    
            elif message["type"] == "clear_history":
                manager.chat_histories[client_id] = []
                await manager.send_personal_message({
                    "type": "chat_update",
                    "data": []
                }, client_id)
                
            elif message["type"] == "ping":
                await manager.send_personal_message({
                    "type": "pong"
                }, client_id)
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(client_id)

# Serve the main chat interface with dramatic emotion and single cloned voice
@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    voice_clone_status = "🎭 Voice Cloning: ENABLED" if AUDIO_PROMPT_PATH else "🎤 Voice Cloning: DISABLED"
    reference_info = f"Reference: {AUDIO_PROMPT_PATH}" if AUDIO_PROMPT_PATH else "Place 'reference_voice.wav' in the same directory to enable voice cloning"
    
    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Natural Emotional Voice Agent</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 20px;
            color: #333;
        }}
        
        .voice-clone-status {{
            background: {'#d4edda' if AUDIO_PROMPT_PATH else '#f8d7da'};
            border: 1px solid {'#c3e6cb' if AUDIO_PROMPT_PATH else '#f5c6cb'};
            color: {'#155724' if AUDIO_PROMPT_PATH else '#721c24'};
            padding: 10px;
            border-radius: 5px;
            margin: 10px auto;
            max-width: 600px;
            text-align: center;
        }}
        
        .feature-badges {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 10px 0;
            flex-wrap: wrap;
        }}
        
        .badge {{
            background: #007bff;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }}
        
        .badge.dramatic {{
            background: #dc3545;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
        
        .chat-container {{
            flex: 1;
            display: flex;
            max-width: 1200px;
            margin: 0 auto;
            gap: 20px;
            height: calc(100vh - 200px);
        }}
        
        .chat-section {{
            flex: 1;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        
        .section-header {{
            background: #007bff;
            color: white;
            padding: 15px;
            font-weight: bold;
            text-align: center;
        }}
        
        .messages {{
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        
        .message {{
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
            animation: slideIn 0.3s ease-out;
            position: relative;
        }}
        
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .message.user {{
            background: #007bff;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }}
        
        .message.assistant {{
            background: #e9ecef;
            color: #333;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
        }}
        
        .input-container {{
            padding: 20px;
            border-top: 1px solid #eee;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        
        .text-input {{
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #ddd;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }}
        
        .text-input:focus {{
            border-color: #007bff;
        }}
        
        .send-btn, .clear-btn {{
            padding: 12px 20px;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
        }}
        
        .send-btn {{
            background: #007bff;
            color: white;
        }}
        
        .send-btn:hover:not(:disabled) {{
            background: #0056b3;
            transform: translateY(-1px);
        }}
        
        .send-btn:disabled {{
            background: #6c757d;
            cursor: not-allowed;
        }}
        
        .clear-btn {{
            background: #dc3545;
            color: white;
        }}
        
        .clear-btn:hover {{
            background: #c82333;
            transform: translateY(-1px);
        }}
        
        .status {{
            padding: 10px 20px;
            background: #f8f9fa;
            border-top: 1px solid #eee;
            text-align: center;
            font-size: 14px;
            color: #6c757d;
        }}
        
        .status.connected {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status.disconnected {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .audio-controls {{
            padding: 15px 20px;
            background: #f8f9fa;
            border-top: 1px solid #eee;
            display: flex;
            align-items: center;
            gap: 15px;
            justify-content: space-between;
        }}
        
        .audio-status {{
            font-size: 14px;
            color: #6c757d;
            font-weight: 500;
        }}
        
        .audio-status.playing {{
            color: #28a745;
            font-weight: bold;
        }}
        
        .audio-status.cloned {{
            color: #6f42c1;
            font-weight: bold;
        }}
        
        .volume-control {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .volume-slider {{
            width: 100px;
        }}
        
        .typing-indicator {{
            display: none;
            font-style: italic;
            color: #6c757d;
            padding: 10px 16px;
        }}
        
        .voice-section {{
            padding: 20px;
            text-align: center;
        }}
        
        .voice-note {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            color: #856404;
        }}
        
        .audio-debug {{
            font-size: 12px;
            color: #6c757d;
            margin-left: 10px;
        }}
        
        .emotion-examples {{
            background: #e7f3ff;
            border: 1px solid #b3d9ff;
            border-radius: 10px;
            padding: 15px;
            margin: 10px auto;
            max-width: 600px;
            text-align: left;
        }}
        
        .emotion-examples h4 {{
            margin-top: 0;
            color: #0066cc;
        }}
        
        .emotion-examples ul {{
            margin: 5px 0;
            padding-left: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎤 Natural Emotional Voice Agent</h1>
        <p>Your cloned voice with genuine emotional conversation responses</p>
        <div class="feature-badges">
            <span class="badge">🎭 Your Cloned Voice</span>
            <span class="badge dramatic">💬 Natural Emotions</span>
            <span class="badge">⚡ Real-time Streaming</span>
        </div>
        <div class="voice-clone-status">
            <strong>{voice_clone_status}</strong><br>
            <small>{reference_info}</small>
        </div>
        <div class="emotion-examples">
            <h4>🎭 Try natural conversations that will evoke emotions:</h4>
            <ul>
                <li><strong>Excitement:</strong> "I just got accepted into my dream university!"</li>
                <li><strong>Sadness:</strong> "My dog passed away yesterday after 12 years together."</li>
                <li><strong>Anger:</strong> "Someone stole my bike that I saved up months to buy."</li>
                <li><strong>Surprise:</strong> "Guess what? I just won the lottery!"</li>
                <li><strong>Confusion:</strong> "Can you explain quantum physics to me?"</li>
                <li><strong>Calm:</strong> "Tell me about a peaceful place you'd like to visit."</li>
            </ul>
        </div>
    </div>
    
    <div class="chat-container">
        <!-- Text Chat Section -->
        <div class="chat-section">
            <div class="section-header">💬 Natural Voice Conversation</div>
            <div class="messages" id="messages"></div>
            <div class="typing-indicator" id="typing">🤖 AI is thinking...</div>
            <div class="input-container">
                <input type="text" class="text-input" id="textInput" placeholder="Type your message here..." maxlength="500">
                <button class="send-btn" id="sendBtn">Send</button>
                <button class="clear-btn" id="clearBtn">Clear</button>
            </div>
            <div class="audio-controls">
                <div>
                    <div class="audio-status" id="audioStatus">🔇 No audio</div>
                    <div class="audio-debug" id="audioDebug">Queue: 0, Playing: false</div>
                </div>
                <div class="volume-control">
                    <label for="volumeSlider">🔊</label>
                    <input type="range" class="volume-slider" id="volumeSlider" min="0" max="1" step="0.1" value="0.7">
                </div>
            </div>
            <div class="status" id="status">Connecting...</div>
        </div>
        
        <!-- Voice Chat Section -->
        <div class="chat-section">
            <div class="section-header">🎤 Voice Chat</div>
            <div class="voice-section">
                <div class="voice-note">
                    <strong>Voice Chat Available!</strong><br>
                    The voice chat interface with dramatic emotional TTS will be mounted when you visit <code>/gradio</code>
                </div>
                <p>For voice input, please visit: <a href="/gradio" target="_blank">/gradio</a></p>
            </div>
        </div>
    </div>

    <script>
        class NaturalEmotionalChatClient {{
            constructor() {{
                this.ws = null;
                this.clientId = this.generateClientId();
                this.audioContext = null;
                this.audioQueue = [];
                this.isPlaying = false;
                this.volume = 0.7;
                this.isConnected = false;
                this.currentSource = null;
                this.currentVoiceType = 'default';
                this.currentEmotion = 'neutral';
                
                this.initializeElements();
                this.setupEventListeners();
                this.connect();
            }}
            
            generateClientId() {{
                return 'client_' + Math.random().toString(36).substr(2, 9);
            }}
            
            initializeElements() {{
                this.elements = {{
                    messages: document.getElementById('messages'),
                    textInput: document.getElementById('textInput'),
                    sendBtn: document.getElementById('sendBtn'),
                    clearBtn: document.getElementById('clearBtn'),
                    status: document.getElementById('status'),
                    audioStatus: document.getElementById('audioStatus'),
                    audioDebug: document.getElementById('audioDebug'),
                    volumeSlider: document.getElementById('volumeSlider'),
                    typing: document.getElementById('typing')
                }};
            }}
            
            setupEventListeners() {{
                this.elements.textInput.addEventListener('keypress', (e) => {{
                    if (e.key === 'Enter' && !e.shiftKey) {{
                        e.preventDefault();
                        this.sendMessage();
                    }}
                }});
                
                this.elements.sendBtn.addEventListener('click', () => this.sendMessage());
                this.elements.clearBtn.addEventListener('click', () => this.clearHistory());
                
                this.elements.volumeSlider.addEventListener('input', (e) => {{
                    this.volume = parseFloat(e.target.value);
                }});
                
                // Auto-reconnect on page visibility change
                document.addEventListener('visibilitychange', () => {{
                    if (!document.hidden && !this.isConnected) {{
                        this.connect();
                    }}
                }});
            }}
            
            getEmotionIcon(emotion) {{
                const icons = {{
                    'excited': '🤩',
                    'happy': '😊',
                    'sad': '😢',
                    'angry': '😠',
                    'surprised': '😲',
                    'confused': '😕',
                    'tired': '😴',
                    'worried': '😟',
                    'calm': '😌',
                    'frustrated': '😤',
                    'enthusiastic': '🎉',
                    'neutral': '😐'
                }};
                return icons[emotion] || '😐';
            }}
            
            connect() {{
                try {{
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${{protocol}}//${{window.location.host}}/ws/${{this.clientId}}`;
                    
                    this.ws = new WebSocket(wsUrl);
                    
                    this.ws.onopen = () => {{
                        this.isConnected = true;
                        this.updateStatus('🟢 Connected', 'connected');
                        this.elements.sendBtn.disabled = false;
                        console.log('WebSocket connected');
                    }};
                    
                    this.ws.onmessage = (event) => {{
                        const message = JSON.parse(event.data);
                        this.handleMessage(message);
                    }};
                    
                    this.ws.onclose = () => {{
                        this.isConnected = false;
                        this.updateStatus('🔴 Disconnected - Reconnecting...', 'disconnected');
                        this.elements.sendBtn.disabled = true;
                        
                        // Auto-reconnect after 3 seconds
                        setTimeout(() => this.connect(), 3000);
                    }};
                    
                    this.ws.onerror = (error) => {{
                        console.error('WebSocket error:', error);
                        this.updateStatus('❌ Connection Error', 'disconnected');
                    }};
                    
                }} catch (error) {{
                    console.error('Failed to connect:', error);
                    this.updateStatus('❌ Failed to Connect', 'disconnected');
                }}
            }}
            
            handleMessage(message) {{
                switch (message.type) {{
                    case 'chat_update':
                        this.updateChatDisplay(message.data);
                        break;
                    case 'audio_start':
                        this.initAudioContext(message.sample_rate);
                        this.stopAllAudio();
                        this.currentVoiceType = message.voice_type || 'default';
                        this.currentEmotion = message.emotion || 'neutral';
                        const emotionIcon = this.getEmotionIcon(this.currentEmotion);
                        const voiceIcon = this.currentVoiceType === 'cloned' ? '🎭' : '🎵';
                        this.updateAudioStatus(`${{voiceIcon}} ${{emotionIcon}} DRAMATIC ${{this.currentEmotion.toUpperCase()}} starting...`, 'playing');
                        break;
                    case 'audio_chunk':
                        this.queueAudioChunk(message.data, message.sample_rate);
                        break;
                    case 'audio_complete':
                        this.updateAudioDebug();
                        setTimeout(() => {{
                            if (!this.isPlaying && this.audioQueue.length === 0) {{
                                const emotionIcon = this.getEmotionIcon(this.currentEmotion);
                                const voiceIcon = this.currentVoiceType === 'cloned' ? '🎭' : '✅';
                                this.updateAudioStatus(`${{voiceIcon}} ${{emotionIcon}} DRAMATIC ${{this.currentEmotion.toUpperCase()}} complete`, '');
                                setTimeout(() => {{
                                    this.updateAudioStatus('🔇 Ready for next dramatic message', '');
                                }}, 2000);
                            }}
                        }}, 500);
                        break;
                    case 'pong':
                        console.log('Received pong');
                        break;
                }}
            }}
            
            async initAudioContext(sampleRate = 22050) {{
                if (!this.audioContext) {{
                    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
                }}
                if (this.audioContext.state === 'suspended') {{
                    await this.audioContext.resume();
                }}
            }}
            
            stopAllAudio() {{
                this.audioQueue = [];
                this.isPlaying = false;
                if (this.currentSource) {{
                    try {{
                        this.currentSource.stop();
                    }} catch (e) {{
                        // Source might already be stopped
                    }}
                    this.currentSource = null;
                }}
                this.updateAudioDebug();
            }}
            
            async queueAudioChunk(base64Audio, sampleRate) {{
                try {{
                    await this.initAudioContext(sampleRate);
                    
                    const arrayBuffer = this.base64ToArrayBuffer(base64Audio);
                    const int16Array = new Int16Array(arrayBuffer);
                    const float32Array = new Float32Array(int16Array.length);
                    
                    for (let i = 0; i < int16Array.length; i++) {{
                        float32Array[i] = int16Array[i] / 32768.0;
                    }}
                    
                    const audioBuffer = this.audioContext.createBuffer(1, float32Array.length, sampleRate);
                    audioBuffer.getChannelData(0).set(float32Array);
                    
                    this.audioQueue.push(audioBuffer);
                    this.updateAudioDebug();
                    
                    if (!this.isPlaying) {{
                        this.playNextAudioChunk();
                    }}
                    
                }} catch (error) {{
                    console.error('Error processing audio chunk:', error);
                    this.updateAudioStatus('❌ Audio error', '');
                }}
            }}
            
            playNextAudioChunk() {{
                if (this.audioQueue.length === 0) {{
                    this.isPlaying = false;
                    this.currentSource = null;
                    this.updateAudioDebug();
                    setTimeout(() => {{
                        if (!this.isPlaying && this.audioQueue.length === 0) {{
                            this.updateAudioStatus('🔇 Audio playback complete', '');
                        }}
                    }}, 100);
                    return;
                }}
                
                this.isPlaying = true;
                const emotionIcon = this.getEmotionIcon(this.currentEmotion);
                const voiceIcon = this.currentVoiceType === 'cloned' ? '🎭' : '🎵';
                const className = this.currentVoiceType === 'cloned' ? 'playing cloned' : 'playing';
                this.updateAudioStatus(`${{voiceIcon}} ${{emotionIcon}} Playing DRAMATIC ${{this.currentEmotion.toUpperCase()}}...`, className);
                
                const audioBuffer = this.audioQueue.shift();
                const source = this.audioContext.createBufferSource();
                const gainNode = this.audioContext.createGain();
                
                this.currentSource = source;
                gainNode.gain.setValueAtTime(this.volume, this.audioContext.currentTime);
                
                source.buffer = audioBuffer;
                source.connect(gainNode);
                gainNode.connect(this.audioContext.destination);
                
                source.onended = () => {{
                    setTimeout(() => {{
                        if (this.currentSource === source) {{
                            this.playNextAudioChunk();
                        }}
                    }}, 5);
                }};
                
                source.start();
                this.updateAudioDebug();
            }}
            
            base64ToArrayBuffer(base64) {{
                const binaryString = window.atob(base64);
                const len = binaryString.length;
                const bytes = new Uint8Array(len);
                for (let i = 0; i < len; i++) {{
                    bytes[i] = binaryString.charCodeAt(i);
                }}
                return bytes.buffer;
            }}
            
            sendMessage() {{
                const text = this.elements.textInput.value.trim();
                if (!text || !this.isConnected) return;
                
                this.stopAllAudio();
                this.updateAudioStatus('🔇 Audio stopped', '');
                
                this.elements.textInput.value = '';
                this.elements.sendBtn.disabled = true;
                this.showTyping(true);
                
                this.ws.send(JSON.stringify({{
                    type: 'text_input',
                    content: text
                }}));
            }}
            
            clearHistory() {{
                if (!this.isConnected) return;
                
                this.stopAllAudio();
                this.ws.send(JSON.stringify({{
                    type: 'clear_history'
                }}));
            }}
            
            updateChatDisplay(chatHistory) {{
                this.elements.messages.innerHTML = '';
                
                chatHistory.forEach((message, index) => {{
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${{message.role}}`;
                    messageDiv.textContent = message.content;
                    
                    if (index === chatHistory.length - 1) {{
                        messageDiv.style.animationDelay = '0.1s';
                    }}
                    
                    this.elements.messages.appendChild(messageDiv);
                }});
                
                this.elements.messages.scrollTop = this.elements.messages.scrollHeight;
                
                const lastMessage = chatHistory[chatHistory.length - 1];
                if (lastMessage && lastMessage.role === 'assistant' && !lastMessage.content.includes('▍')) {{
                    this.elements.sendBtn.disabled = false;
                    this.showTyping(false);
                }}
            }}
            
            showTyping(show) {{
                this.elements.typing.style.display = show ? 'block' : 'none';
            }}
            
            updateStatus(text, className = '') {{
                this.elements.status.textContent = text;
                this.elements.status.className = `status ${{className}}`;
            }}
            
            updateAudioStatus(text, className = '') {{
                this.elements.audioStatus.textContent = text;
                this.elements.audioStatus.className = `audio-status ${{className}}`;
            }}
            
            updateAudioDebug() {{
                if (this.elements.audioDebug) {{
                    const emotionIcon = this.getEmotionIcon(this.currentEmotion);
                    const voiceType = this.currentVoiceType === 'cloned' ? '🎭 ' : '';
                    this.elements.audioDebug.textContent = `${{voiceType}}${{emotionIcon}} ${{this.currentEmotion.toUpperCase()}} Queue: ${{this.audioQueue.length}}, Playing: ${{this.isPlaying}}`;
                }}
            }}
        }}
        
        // Initialize the chat client when the page loads
        document.addEventListener('DOMContentLoaded', () => {{
            new NaturalEmotionalChatClient();
        }});
    </script>
</body>
</html>
    """)

# Voice chat handler with dramatic emotion and single cloned voice
async def voice_chat_handler_async(audio: tuple[int, np.ndarray], chat_history_ui: list | None = None):
    chat_history_ui = chat_history_ui or []
    prompt_text = stt_model.stt(audio)
    if not prompt_text or not prompt_text.strip():
        yield AdditionalOutputs(chat_history_ui)
        return

    # Use the same core streaming function but for Gradio
    current_chat_history = chat_history_ui
    current_chat_history.append({"role": "user", "content": prompt_text})
    yield AdditionalOutputs(current_chat_history)

    llm_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg_obj in list(current_chat_history):
        if msg_obj["role"] == "user": 
            llm_messages.append(HumanMessage(content=msg_obj["content"]))
        elif msg_obj["role"] == "assistant": 
            llm_messages.append(AIMessage(content=msg_obj["content"]))
    
    full_llm_response_text = ""
    assistant_message_index = len(current_chat_history)
    current_chat_history.append({"role": "assistant", "content": "▍"})
    yield AdditionalOutputs(current_chat_history)

    llm_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    llm_thread_task = loop.create_task(
        asyncio.to_thread(_threaded_llm_streamer, llm_messages, llm_queue, llm)
    )

    while True:
        llm_chunk = await llm_queue.get()
        if llm_chunk is None:
            llm_queue.task_done()
            break
        chunk_content = getattr(llm_chunk, 'content', '')
        if chunk_content:
            full_llm_response_text += chunk_content
            current_chat_history[assistant_message_index]["content"] = full_llm_response_text + "▍"
            yield AdditionalOutputs(current_chat_history)
        llm_queue.task_done()
    
    await llm_thread_task

    current_chat_history[assistant_message_index]["content"] = full_llm_response_text
    yield AdditionalOutputs(current_chat_history)

    # TTS with dramatic emotion and single cloned voice for Gradio
    if full_llm_response_text.strip():
        # Detect emotion from the response
        detected_emotion = detect_emotion_from_text(full_llm_response_text, debug=True)
        
        tts_queue = asyncio.Queue()
        tts_thread_task = loop.create_task(
            asyncio.to_thread(
                _threaded_tts_generator_with_natural_emotion, 
                full_llm_response_text, 
                tts_queue, 
                tts_model, 
                0.15,  # chunk duration
                AUDIO_PROMPT_PATH,  # voice cloning reference
                detected_emotion    # detected emotion with dramatic parameters
            )
        )
        while True:
            audio_chunk_tensor = await tts_queue.get()
            if audio_chunk_tensor is None:
                tts_queue.task_done()
                break
            processed_audio_1d = convert_tensor_audio_to_int16(audio_chunk_tensor)
            if processed_audio_1d.size > 0:
                yield tts_model.sr, processed_audio_1d.reshape(1, -1)
            tts_queue.task_done()
        await tts_thread_task

# Gradio interface for voice chat with natural emotion and single cloned voice
voice_clone_title = f"🎤 Natural Emotional Voice Chat ({'CLONED' if AUDIO_PROMPT_PATH else 'DEFAULT'} Voice)"
shared_chatbot_display = gr.Chatbot(
    label="Natural Emotional Voice Conversation", 
    type="messages", 
    bubble_full_width=False, 
    height=500, 
    show_label=False
)

stream = Stream(
    handler=ReplyOnPause(voice_chat_handler_async, input_sample_rate=S3_SR),
    modality="audio", 
    mode="send-receive",
    additional_inputs=[shared_chatbot_display], 
    additional_outputs=[shared_chatbot_display],
    additional_outputs_handler=lambda _, nv: nv,
    ui_args={"title": voice_clone_title}
)

# Mount Gradio app for voice interface
app = gr.mount_gradio_app(app, stream.ui, path="/gradio")

def main():
    import uvicorn
    import os
    os.environ["GRADIO_SSR_MODE"] = "false"
    try: 
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError: 
        nltk.download('punkt', quiet=True)
    
    print("🚀 Starting Natural Emotional Voice Agent...")
    print("📱 Text Chat (Natural Emotions): http://localhost:8000/")
    print("🎤 Voice Chat (Gradio): http://localhost:8000/gradio")
    
    if AUDIO_PROMPT_PATH:
        print(f"🎭 Voice Cloning: ENABLED with reference '{AUDIO_PROMPT_PATH}'")
        print("🎙️ Using CLONED voice with natural emotional conversation!")
    else:
        print("🎤 Voice Cloning: DISABLED - place 'reference_voice.wav' in the directory")
        print("🎙️ Using default voice with natural emotional conversation!")
    
    print("🎭 Natural Emotion Detection: Responds genuinely to conversational context")
    print("📊 Emotion ranges: exaggeration(0.05-0.95), cfg_weight(0.2-0.95), temperature(0.3-1.3)")
    print("💬 Features: Natural conversation + Emotional voice + Real-time streaming!")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()