import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import tempfile
import librosa
import soundfile as sf
from transformers import ClapProcessor, ClapModel, WhisperProcessor, WhisperModel, AutoTokenizer, AutoModel
import torch
import psutil
import threading
from datetime import datetime
import re

st.set_page_config(page_title="Text-to-Audio Search", layout="wide")

# ============================================================================
# MODEL CONFIGURATION - Easy to swap different models here!
# ============================================================================

AVAILABLE_MODELS = {
    "CLAP-HTSAT (Recommended)": {
        "model_name": "laion/clap-htsat-unfused",
        "description": "General purpose audio-text alignment, good balance of speed/accuracy"
    },
    "CLAP-HTSAT-Fused": {
        "model_name": "laion/clap-htsat-fused", 
        "description": "Fused version, potentially better accuracy but slower"
    },
    "Microsoft CLAP": {
        "model_name": "microsoft/clap-htsat-unfused",
        "description": "Microsoft's version, good for speech and general audio"
    }
}

# ASR Models for speech embedding
ASR_MODELS = {
    "Whisper-Base": {
        "model_name": "openai/whisper-base",
        "description": "Good balance of accuracy and speed for transcription"
    },
    "Whisper-Small": {
        "model_name": "openai/whisper-small", 
        "description": "Better accuracy, slower processing"
    }
}

# Text embedding models for ASR text processing
TEXT_EMBED_MODELS = {
    "BERT-Base": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "description": "High quality sentence embeddings"
    },
    "MiniLM": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Faster, smaller model"
    }
}

# Fusion strategies
FUSION_STRATEGIES = {
    "Fixed 50/50": "Combine audio and ASR embeddings with equal 50/50 weights",
    "Dynamic Selection": "Choose either audio OR ASR embedding based on query type",
    "Adaptive Weighting": "Dynamically weight audio vs ASR based on query analysis"
}

# Session state for metrics
if 'metrics_log' not in st.session_state:
    st.session_state.metrics_log = []
if 'audio_database' not in st.session_state:
    st.session_state.audio_database = []
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'enable_multimodal' not in st.session_state:
    st.session_state.enable_multimodal = False
if 'asr_model_loaded' not in st.session_state:
    st.session_state.asr_model_loaded = False

def log_metric(operation, duration, details=None):
    """Log performance metrics with timestamp"""
    metric = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "operation": operation,
        "duration_ms": round(duration * 1000, 2),
        "duration_s": round(duration, 3),
        "details": details or {}
    }
    st.session_state.metrics_log.append(metric)
    return metric

def get_system_metrics():
    """Get current system resource usage"""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_gb": round(psutil.virtual_memory().used / (1024**3), 2)
    }

st.title("🎵 Text-to-Audio Search Demo")
st.write("Upload audio files and search for them using natural language descriptions - no transcription needed!")

# Sidebar for model selection and settings
st.sidebar.header("🔧 Model & Settings")

# Search mode selection
search_mode = st.sidebar.selectbox(
    "🚀 Search Mode:",
    ["Full CLAP (Best Quality)", "Lightweight (Fast & Small)"],
    help="Choose between full-featured CLAP models or lightweight sentence-transformer approach"
)

if search_mode == "Lightweight (Fast & Small)":
    st.info("💡 **Lightweight Mode Available!**")
    st.write("A prototype lightweight system using:")
    st.write("- Whisper-tiny (39MB) for speech-to-text")
    st.write("- MiniLM-L6-v2 (23MB) for text embeddings") 
    st.write("- Audio feature bridge (<1MB) for unified embedding space")
    st.write("- **Total: ~60MB vs 400MB+ for CLAP**")
    
    if st.button("🚀 Try Lightweight Prototype"):
        st.success("Opening lightweight demo...")
        st.code("streamlit run lightweight_audio_search.py --server.port 8504")
        st.write("Run the above command in terminal to try the lightweight system!")
    
    st.info("ℹ️ Current page continues with full CLAP system below.")

selected_model_name = st.sidebar.selectbox(
    "Choose CLAP Model:",
    list(AVAILABLE_MODELS.keys()),
    help="Different models have different strengths. Try them all!"
)

selected_model_config = AVAILABLE_MODELS[selected_model_name]
st.sidebar.write(f"**Description:** {selected_model_config['description']}")

# Multi-modal settings
st.sidebar.subheader("🎭 Multi-Modal Search")
enable_multimodal = st.sidebar.checkbox(
    "Enable ASR + Audio Fusion", 
    value=st.session_state.enable_multimodal,
    help="Combine raw audio embeddings with speech transcription embeddings"
)

if enable_multimodal != st.session_state.enable_multimodal:
    st.session_state.enable_multimodal = enable_multimodal
    if enable_multimodal:
        st.sidebar.info("⚡ Multi-modal mode enabled! This will add ASR processing.")

fusion_strategy = None
asr_model_name = None
text_model_name = None

if enable_multimodal:
    fusion_strategy = st.sidebar.selectbox(
        "Fusion Strategy:",
        list(FUSION_STRATEGIES.keys()),
        help="How to combine audio and ASR embeddings"
    )
    st.sidebar.write(f"📝 {FUSION_STRATEGIES[fusion_strategy]}")
    
    asr_model_name = st.sidebar.selectbox(
        "ASR Model:",
        list(ASR_MODELS.keys()),
        help="Model for speech transcription"
    )
    
    text_model_name = st.sidebar.selectbox(
        "Text Embedding Model:",
        list(TEXT_EMBED_MODELS.keys()),
        help="Model for embedding transcribed text"
    )

# Initialize session state for storing embeddings
if 'audio_database' not in st.session_state:
    st.session_state.audio_database = []

@st.cache_resource
def load_audio_model(model_name):
    """Load CLAP model for audio embeddings with detailed metrics"""
    load_start = time.time()
    system_before = get_system_metrics()
    
    with st.spinner(f"Loading {model_name}..."):
        model_start = time.time()
        model = ClapModel.from_pretrained(model_name)
        model_load_time = time.time() - model_start
        
        processor_start = time.time()
        processor = ClapProcessor.from_pretrained(model_name)
        processor_load_time = time.time() - processor_start
    
    total_load_time = time.time() - load_start
    system_after = get_system_metrics()
    
    # Calculate model size approximation
    model_params = sum(p.numel() for p in model.parameters())
    model_size_mb = (model_params * 4) / (1024 * 1024)  # Assuming float32
    
    details = {
        "model_name": model_name,
        "model_load_time_s": round(model_load_time, 3),
        "processor_load_time_s": round(processor_load_time, 3),
        "model_params": f"{model_params:,}",
        "estimated_size_mb": round(model_size_mb, 1),
        "memory_increase_mb": round((system_after["memory_used_gb"] - system_before["memory_used_gb"]) * 1024, 1),
        "cpu_usage": f"{system_after['cpu_percent']}%"
    }
    
    log_metric("Model Loading", total_load_time, details)
    
    st.session_state.current_model = model_name
    return model, processor

@st.cache_resource
def load_asr_models(asr_model_name, text_model_name):
    """Load ASR and text embedding models for speech processing"""
    load_start = time.time()
    
    with st.spinner(f"Loading ASR models ({asr_model_name} + {text_model_name})..."):
        # Load Whisper for transcription
        asr_start = time.time()
        whisper_model = WhisperModel.from_pretrained(asr_model_name)
        whisper_processor = WhisperProcessor.from_pretrained(asr_model_name)
        asr_load_time = time.time() - asr_start
        
        # Load text embedding model
        text_start = time.time()
        from sentence_transformers import SentenceTransformer
        text_embedder = SentenceTransformer(text_model_name)
        text_load_time = time.time() - text_start
    
    total_load_time = time.time() - load_start
    
    details = {
        "asr_model": asr_model_name,
        "text_model": text_model_name,
        "asr_load_time_s": round(asr_load_time, 3),
        "text_load_time_s": round(text_load_time, 3),
        "combined_load_time_s": round(total_load_time, 3)
    }
    
    log_metric("ASR Models Loading", total_load_time, details)
    return whisper_model, whisper_processor, text_embedder

def analyze_query_type_basic(query_text):
    """
    Basic keyword-based query analysis (fallback method)
    """
    query_lower = query_text.lower()
    
    # Speech-related keywords
    speech_keywords = [
        'speech', 'talk', 'voice', 'speak', 'word', 'conversation', 'dialogue',
        'interview', 'discussion', 'narrator', 'announcement', 'commentary'
    ]
    
    # Audio/music keywords  
    audio_keywords = [
        'music', 'song', 'melody', 'instrument', 'drum', 'guitar', 'piano',
        'sound', 'noise', 'ambient', 'nature', 'animal', 'effect'
    ]
    
    speech_score = sum(1 for keyword in speech_keywords if keyword in query_lower)
    audio_score = sum(1 for keyword in audio_keywords if keyword in query_lower)
    
    if speech_score > audio_score:
        confidence = speech_score / (speech_score + audio_score + 1)
        return True, confidence, f"Keyword analysis: {speech_score} speech vs {audio_score} audio keywords"
    else:
        confidence = (audio_score + 1) / (speech_score + audio_score + 1)  # +1 for audio bias
        return False, confidence, f"Keyword analysis: {speech_score} speech vs {audio_score} audio keywords"

def analyze_query_type_advanced(query_text, text_embedder):
    """
    Advanced query analysis using semantic similarity to determine speech vs audio focus
    Returns: (is_speech_related: bool, confidence: float, reasoning: str, scores: dict)
    """
    # Fallback to basic keyword method if no text embedder provided
    if text_embedder is None:
        return analyze_query_type_basic(query_text)
    
    query_lower = query_text.lower()
    
    # Define prototype queries for each category
    speech_prototypes = [
        "person talking and speaking",
        "human voice conversation dialogue",
        "speech and verbal communication",
        "someone saying words and phrases",
        "interview discussion presentation",
        "narrator announcer commentary voice"
    ]
    
    audio_prototypes = [
        "musical instruments and melodies",
        "sound effects and ambient noise",
        "nature sounds and environmental audio", 
        "electronic music and synthesized sounds",
        "drums guitar piano instrumental music",
        "mechanical sounds and audio textures"
    ]
    
    try:
        # Get embeddings for query and prototypes
        query_embedding = text_embedder.encode([query_text])[0]
        speech_embeddings = text_embedder.encode(speech_prototypes)
        audio_embeddings = text_embedder.encode(audio_prototypes)
        
        # Calculate similarities
        speech_similarities = cosine_similarity([query_embedding], speech_embeddings)[0]
        audio_similarities = cosine_similarity([query_embedding], audio_embeddings)[0]
        
        # Get max similarities for each category
        max_speech_sim = float(np.max(speech_similarities))
        max_audio_sim = float(np.max(audio_similarities))
        
        # Also do keyword-based analysis as backup
        keyword_is_speech, keyword_conf, keyword_reasoning = analyze_query_type_keyword(query_text)
        
        # Combine semantic and keyword approaches
        semantic_weight = 0.7
        keyword_weight = 0.3
        
        if max_speech_sim > max_audio_sim:
            semantic_is_speech = True
            semantic_confidence = max_speech_sim / (max_speech_sim + max_audio_sim) if (max_speech_sim + max_audio_sim) > 0 else 0.5
        else:
            semantic_is_speech = False
            semantic_confidence = max_audio_sim / (max_speech_sim + max_audio_sim) if (max_speech_sim + max_audio_sim) > 0 else 0.5
        
        # Combined decision
        if semantic_is_speech == keyword_is_speech:
            # Both methods agree
            final_is_speech = semantic_is_speech
            final_confidence = semantic_weight * semantic_confidence + keyword_weight * keyword_conf
            agreement = "Both semantic and keyword analysis agree"
        else:
            # Methods disagree, prefer semantic but reduce confidence
            final_is_speech = semantic_is_speech
            final_confidence = semantic_weight * semantic_confidence * 0.8  # Reduce confidence due to disagreement
            agreement = f"Disagreement: semantic says {'speech' if semantic_is_speech else 'audio'}, keywords say {'speech' if keyword_is_speech else 'audio'}"
        
        reasoning = f"Semantic analysis: speech={max_speech_sim:.3f}, audio={max_audio_sim:.3f}. {agreement}. Keyword: {keyword_reasoning}"
        
        scores = {
            "semantic_speech_score": max_speech_sim,
            "semantic_audio_score": max_audio_sim,
            "keyword_speech_confidence": keyword_conf if keyword_is_speech else 1-keyword_conf,
            "keyword_audio_confidence": keyword_conf if not keyword_is_speech else 1-keyword_conf,
            "final_confidence": final_confidence,
            "agreement": semantic_is_speech == keyword_is_speech
        }
        
        return final_is_speech, final_confidence, reasoning, scores
        
    except Exception as e:
        # Fallback to keyword-only analysis
        return analyze_query_type_keyword(query_text) + ({},)

def analyze_query_type_keyword(query_text):
    """
    Fallback keyword-based analysis (original method)
    Returns: (is_speech_related: bool, confidence: float, reasoning: str)
    """
    query_lower = query_text.lower()
    
    # Speech-related keywords and patterns
    speech_keywords = [
        'speech', 'talking', 'conversation', 'dialogue', 'voice', 'speaking',
        'words', 'language', 'accent', 'pronunciation', 'verbal', 'oral',
        'interview', 'lecture', 'presentation', 'discussion', 'monologue',
        'narrator', 'announcer', 'commentary', 'news', 'podcast'
    ]
    
    # Phrase patterns that suggest speech content
    speech_phrases = [
        r'someone (saying|talking|speaking)',
        r'person (saying|talking|speaking)',
        r'man (saying|talking|speaking)',
        r'woman (saying|talking|speaking)',
        r'(says?|said|tell|telling|speak|speaking|talk|talking)',
        r'(phrase|sentence|word|words) ".*"',
        r'in (english|spanish|french|german|chinese|japanese)',
        r'with (accent|pronunciation)'
    ]
    
    # Audio/music keywords
    audio_keywords = [
        'music', 'song', 'melody', 'rhythm', 'beat', 'instrument', 'sound effect',
        'noise', 'ambient', 'nature', 'animal', 'mechanical', 'electronic',
        'piano', 'guitar', 'drum', 'violin', 'synthesizer', 'bass',
        'bird', 'water', 'wind', 'rain', 'engine', 'door', 'footsteps'
    ]
    
    # Count matches
    speech_score = 0
    audio_score = 0
    reasoning_parts = []
    
    # Check direct keyword matches
    for keyword in speech_keywords:
        if keyword in query_lower:
            speech_score += 1
            reasoning_parts.append(f"speech keyword: '{keyword}'")
    
    for keyword in audio_keywords:
        if keyword in query_lower:
            audio_score += 1
            reasoning_parts.append(f"audio keyword: '{keyword}'")
    
    # Check phrase patterns
    for pattern in speech_phrases:
        if re.search(pattern, query_lower):
            speech_score += 2  # Phrase matches are stronger indicators
            reasoning_parts.append(f"speech pattern: '{pattern}'")
    
    # Calculate confidence and decision
    total_score = speech_score + audio_score
    if total_score == 0:
        # No clear indicators, default to audio (more general)
        is_speech_related = False
        confidence = 0.5
        reasoning = "No clear speech/audio indicators, defaulting to audio"
    else:
        is_speech_related = speech_score > audio_score
        confidence = max(speech_score, audio_score) / total_score
        reasoning = "; ".join(reasoning_parts)
    
    return is_speech_related, confidence, reasoning

def generate_adaptive_weights_advanced(query_text, text_embedder=None):
    """
    Enhanced adaptive weight generation using semantic analysis
    Returns: (audio_weight: float, asr_weight: float, explanation: str, analysis_details: dict)
    """
    if text_embedder:
        is_speech, confidence, reasoning, scores = analyze_query_type_advanced(query_text, text_embedder)
    else:
        is_speech, confidence, reasoning = analyze_query_type_keyword(query_text)
        scores = {}
    
    if is_speech:
        # Speech-related query: favor ASR embeddings
        base_asr_weight = 0.7
        base_audio_weight = 0.3
        
        # Adjust based on confidence (more confident = more extreme weighting)
        confidence_boost = (confidence - 0.5) * 0.4  # Scale confidence to [-0.2, 0.2]
        asr_weight = np.clip(base_asr_weight + confidence_boost, 0.1, 0.9)
        audio_weight = 1.0 - asr_weight
        
        explanation = f"Speech-focused (conf: {confidence:.2f}): ASR {asr_weight:.1%}, Audio {audio_weight:.1%}"
    else:
        # Audio/music query: favor raw audio embeddings
        base_audio_weight = 0.7
        base_asr_weight = 0.3
        
        # Adjust based on confidence  
        confidence_boost = (confidence - 0.5) * 0.4
        audio_weight = np.clip(base_audio_weight + confidence_boost, 0.1, 0.9)
        asr_weight = 1.0 - audio_weight
        
        explanation = f"Audio-focused (conf: {confidence:.2f}): Audio {audio_weight:.1%}, ASR {asr_weight:.1%}"
    
    analysis_details = {
        "query_type": "speech" if is_speech else "audio",
        "confidence": confidence,
        "reasoning": reasoning,
        "scores": scores,
        "audio_weight": audio_weight,
        "asr_weight": asr_weight
    }
    
    return audio_weight, asr_weight, explanation, analysis_details

def get_asr_embedding(audio_chunk, whisper_model, whisper_processor, text_embedder, sample_rate=48000):
    """Get ASR-based embedding: transcribe audio then embed the text"""
    embed_start = time.time()
    
    try:
        # Transcribe audio using Whisper
        transcribe_start = time.time()
        
        # Prepare audio for Whisper
        audio_input = whisper_processor(audio_chunk, sampling_rate=sample_rate, return_tensors="pt")
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = whisper_model.generate(audio_input["input_features"])
            transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        transcribe_time = time.time() - transcribe_start
        
        # Embed the transcribed text
        text_embed_start = time.time()
        if transcription.strip():
            text_embedding = text_embedder.encode([transcription])[0]
        else:
            # No speech detected, return zero embedding
            text_embedding = np.zeros(text_embedder.get_sentence_embedding_dimension())
        
        text_embed_time = time.time() - text_embed_start
        
    except Exception as e:
        # Fallback: return zero embedding if ASR fails
        transcription = "[ASR_FAILED]"
        text_embedding = np.zeros(384)  # Default dimension
        transcribe_time = 0
        text_embed_time = 0
    
    total_time = time.time() - embed_start
    
    details = {
        "transcription": transcription[:100] + "..." if len(transcription) > 100 else transcription,
        "transcription_length": len(transcription),
        "transcribe_time_s": round(transcribe_time, 3),
        "text_embed_time_s": round(text_embed_time, 3),
        "has_speech": len(transcription.strip()) > 0,
        "embedding_dim": len(text_embedding)
    }
    
    log_metric("ASR Embedding", total_time, details)
    
    return text_embedding, transcription
    """Load CLAP model for audio embeddings with detailed metrics"""
    load_start = time.time()
    system_before = get_system_metrics()
    
    with st.spinner(f"Loading {model_name}..."):
        model_start = time.time()
        model = ClapModel.from_pretrained(model_name)
        model_load_time = time.time() - model_start
        
        processor_start = time.time()
        processor = ClapProcessor.from_pretrained(model_name)
        processor_load_time = time.time() - processor_start
    
    total_load_time = time.time() - load_start
    system_after = get_system_metrics()
    
    # Calculate model size approximation
    model_params = sum(p.numel() for p in model.parameters())
    model_size_mb = (model_params * 4) / (1024 * 1024)  # Assuming float32
    
    details = {
        "model_name": model_name,
        "model_load_time_s": round(model_load_time, 3),
        "processor_load_time_s": round(processor_load_time, 3),
        "model_params": f"{model_params:,}",
        "estimated_size_mb": round(model_size_mb, 1),
        "memory_increase_mb": round((system_after["memory_used_gb"] - system_before["memory_used_gb"]) * 1024, 1),
        "cpu_usage": f"{system_after['cpu_percent']}%"
    }
    
    log_metric("Model Loading", total_load_time, details)
    
    st.session_state.current_model = model_name
    return model, processor

def load_and_process_audio(file_path, target_sr=48000, chunk_duration=10):
    """
    Load audio file and split into chunks with metrics
    Returns list of audio chunks and their start times
    """
    processing_start = time.time()
    
    audio, sr = librosa.load(file_path, sr=target_sr)
    load_time = time.time() - processing_start
    
    # Split into chunks
    chunk_start = time.time()
    chunk_samples = int(chunk_duration * target_sr)
    chunks = []
    start_times = []
    
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) >= target_sr:  # Only keep chunks >= 1 second
            chunks.append(chunk)
            start_times.append(i / target_sr)
    
    chunk_time = time.time() - chunk_start
    total_time = time.time() - processing_start
    
    details = {
        "file_duration_s": round(len(audio) / target_sr, 1),
        "chunks_created": len(chunks),
        "audio_load_time_s": round(load_time, 3),
        "chunking_time_s": round(chunk_time, 3),
        "sample_rate": target_sr
    }
    
    log_metric("Audio Processing", total_time, details)
    
    return chunks, start_times, sr

def get_audio_embedding(audio_chunk, model, processor, sample_rate=48000):
    """Get embedding for a single audio chunk with metrics"""
    embed_start = time.time()
    
    with torch.no_grad():
        inputs = processor(audio=audio_chunk, sampling_rate=sample_rate, return_tensors="pt")
        audio_embed = model.get_audio_features(**inputs)
        embedding = audio_embed.cpu().numpy().flatten()
    
    embed_time = time.time() - embed_start
    
    details = {
        "embedding_dim": len(embedding),
        "chunk_duration_s": round(len(audio_chunk) / sample_rate, 1),
        "embedding_size_kb": round(embedding.nbytes / 1024, 2)
    }
    
    log_metric("Audio Embedding", embed_time, details)
    
    return embedding

def get_text_embedding(text, model, processor):
    """Get embedding for text query with metrics - uses CLAP's text encoder"""
    embed_start = time.time()
    
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt")
        text_embed = model.get_text_features(**inputs)
        embedding = text_embed.cpu().numpy().flatten()
    
    embed_time = time.time() - embed_start
    
    details = {
        "text_length": len(text),
        "embedding_dim": len(embedding),
        "embedding_size_kb": round(embedding.nbytes / 1024, 2)
    }
    
    log_metric("Text Embedding", embed_time, details)
    
    return embedding

def save_audio_chunk(audio_chunk, sample_rate, filename):
    """Save audio chunk to temp file for playback"""
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)
    sf.write(filepath, audio_chunk, sample_rate)
    return filepath

def run_search_strategy(query_text, strategy, model, processor, asr_models, audio_database, enable_multimodal):
    """Run a specific search strategy and return results"""
    start_time = time.time()
    
    # Get text embedding using CLAP's text encoder
    query_embedding = get_text_embedding(query_text, model, processor)
    
    similarities = []
    
    # Check if database has multi-modal data
    has_multimodal = any(item.get('is_multimodal', False) for item in audio_database)
    
    if not enable_multimodal or not has_multimodal or strategy == "Audio Only":
        # Pure audio search
        for item in audio_database:
            sim = cosine_similarity([query_embedding], [item['embedding']])[0][0]
            similarities.append(sim)
        fusion_info = {"strategy": "Audio Only", "details": "Direct CLAP audio-text similarity"}
    
    elif strategy == "Fixed 50/50":
        for item in audio_database:
            # Audio similarity
            audio_sim = cosine_similarity([query_embedding], [item['embedding']])[0][0]
            
            # ASR similarity (if available and dimensions match)
            if item.get('asr_embedding') is not None:
                if len(query_embedding) == len(item['asr_embedding']):
                    asr_sim = cosine_similarity([query_embedding], [item['asr_embedding']])[0][0]
                    # Fixed 50/50 combination
                    combined_sim = 0.5 * audio_sim + 0.5 * asr_sim
                else:
                    # Dimension mismatch - use audio only
                    combined_sim = audio_sim
            else:
                # Fallback to audio only if no ASR
                combined_sim = audio_sim
            
            similarities.append(combined_sim)
        fusion_info = {"strategy": "Fixed 50/50", "audio_weight": 0.5, "asr_weight": 0.5}
    
    elif strategy == "Dynamic Selection":
        is_speech, confidence, reasoning = analyze_query_type_advanced(query_text, asr_models['text_embedder'])
        if is_speech:
            # Use ASR embeddings only
            for item in audio_database:
                if item.get('asr_embedding') is not None:
                    # Check if dimensions match before computing similarity
                    if len(query_embedding) == len(item['asr_embedding']):
                        sim = cosine_similarity([query_embedding], [item['asr_embedding']])[0][0]
                    else:
                        # Dimension mismatch - fallback to audio embedding
                        sim = cosine_similarity([query_embedding], [item['embedding']])[0][0]
                else:
                    # Fallback to audio if no ASR
                    sim = cosine_similarity([query_embedding], [item['embedding']])[0][0]
                similarities.append(sim)
            fusion_info = {"strategy": "Dynamic Selection", "selected": "ASR", "confidence": confidence, "reasoning": reasoning}
        else:
            # Use audio embeddings only
            for item in audio_database:
                sim = cosine_similarity([query_embedding], [item['embedding']])[0][0]
                similarities.append(sim)
            fusion_info = {"strategy": "Dynamic Selection", "selected": "Audio", "confidence": confidence, "reasoning": reasoning}
    
    elif strategy == "Adaptive Weighting":
        audio_weight, asr_weight, explanation, _ = generate_adaptive_weights_advanced(query_text)
        for item in audio_database:
            # Audio similarity
            audio_sim = cosine_similarity([query_embedding], [item['embedding']])[0][0]
            
            # ASR similarity (if available and dimensions match)
            if item.get('asr_embedding') is not None:
                if len(query_embedding) == len(item['asr_embedding']):
                    asr_sim = cosine_similarity([query_embedding], [item['asr_embedding']])[0][0]
                    # Adaptive combination
                    combined_sim = audio_weight * audio_sim + asr_weight * asr_sim
                else:
                    # Dimension mismatch - use audio only
                    combined_sim = audio_sim
            else:
                # Fallback to audio only if no ASR
                combined_sim = audio_sim
            
            similarities.append(combined_sim)
        fusion_info = {"strategy": "Adaptive Weighting", "audio_weight": audio_weight, "asr_weight": asr_weight, "explanation": explanation}
    
    retrieval_time = time.time() - start_time
    return np.array(similarities), fusion_info, retrieval_time

def display_comparison_results(query_text, all_results, all_fusion_info, top_k):
    """Display side-by-side comparison of all strategies"""
    st.markdown("### 🎯 Query Analysis")
    st.write(f"**Query:** '{query_text}'")
    
    # Show query analysis
    is_speech, confidence, reasoning = analyze_query_type_advanced(query_text, None)  # Fallback to basic method for display
    st.write(f"**Query Type:** {'Speech-focused' if is_speech else 'Audio-focused'} (confidence: {confidence:.3f})")
    st.write(f"**Analysis:** {reasoning}")
    
    st.markdown("### 📊 Results Comparison")
    
    # Create columns for each strategy
    strategies = list(all_results.keys())
    cols = st.columns(len(strategies))
    
    for i, strategy in enumerate(strategies):
        with cols[i]:
            st.markdown(f"**{strategy}**")
            
            # Show fusion info
            fusion_info = all_fusion_info[strategy]
            with st.expander("Strategy Details", expanded=False):
                if "audio_weight" in fusion_info:
                    st.write(f"Audio weight: {fusion_info['audio_weight']:.3f}")
                if "asr_weight" in fusion_info:
                    st.write(f"ASR weight: {fusion_info['asr_weight']:.3f}")
                if "explanation" in fusion_info:
                    st.write(f"Reasoning: {fusion_info['explanation']}")
                if "selected" in fusion_info:
                    st.write(f"Selected: {fusion_info['selected']}")
            
            # Show top results
            similarities = all_results[strategy]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for rank, idx in enumerate(top_indices, 1):
                similarity = similarities[idx]
                audio_info = st.session_state.audio_database[idx]
                filename = audio_info['filename']
                
                st.markdown(f"**{rank}.** {filename}")
                st.markdown(f"Score: {similarity:.4f}")
                
                # Audio player
                audio_path = audio_info['filepath']
                try:
                    audio_file = open(audio_path, 'rb')
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav')
                    audio_file.close()
                except:
                    st.error(f"Could not load audio: {filename}")
                
                st.markdown("---")

def display_single_strategy_results(query_text, similarities, fusion_info, top_k, strategy_name):
    """Display results for a single strategy"""
    # Show query analysis if multimodal
    if fusion_info["strategy"] != "Audio Only":
        st.subheader("🎯 Query Analysis")
        if "reasoning" in fusion_info:
            st.write(f"**Analysis:** {fusion_info['reasoning']}")
        if "explanation" in fusion_info:
            st.write(f"**Strategy Logic:** {fusion_info['explanation']}")
    
    # Show fusion details
    st.subheader("🔧 Fusion Details")
    if "audio_weight" in fusion_info and "asr_weight" in fusion_info:
        st.write(f"Audio weight: {fusion_info['audio_weight']:.3f}, ASR weight: {fusion_info['asr_weight']:.3f}")
    elif "selected" in fusion_info:
        st.write(f"Selected pipeline: {fusion_info['selected']}")
    else:
        st.write("Using direct audio-text similarity")
    
    # Display results
    st.subheader(f"🎵 Top {top_k} Results")
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    for rank, idx in enumerate(top_indices, 1):
        similarity = similarities[idx]
        audio_info = st.session_state.audio_database[idx]
        filename = audio_info['filename']
        
        # Create expandable result
        with st.expander(f"{rank}. {filename} (Score: {similarity:.4f})", expanded=(rank <= 3)):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Audio player
                audio_path = audio_info['filepath']
                try:
                    audio_file = open(audio_path, 'rb')
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav')
                    audio_file.close()
                except:
                    st.error(f"Could not load audio: {filename}")
            
            with col2:
                st.metric("Similarity Score", f"{similarity:.4f}")
                if 'transcription' in audio_info and audio_info['transcription']:
                    st.write("**Transcription:**")
                    st.write(audio_info['transcription'][:200] + "..." if len(audio_info['transcription']) > 200 else audio_info['transcription'])

# Legacy wrapper functions for backward compatibility
def analyze_query_type(query_text):
    """Wrapper for backward compatibility"""
    return analyze_query_type_advanced(query_text, None)  # Use None to trigger fallback

def generate_adaptive_weights(query_text):
    """Generate adaptive weights based on query analysis - simplified version"""
    audio_weight, asr_weight, explanation, _ = generate_adaptive_weights_advanced(query_text)
    return audio_weight, asr_weight, explanation


# Load models (will reload if user changes model selection)
model_changed = (st.session_state.current_model != selected_model_config["model_name"])
if model_changed:
    st.cache_resource.clear()  # Clear cache to reload with new model

with st.spinner("Loading AI model..." if model_changed else "Model ready"):
    model, processor = load_audio_model(selected_model_config["model_name"])

# Load ASR models if multi-modal is enabled
asr_models = None
if enable_multimodal:
    asr_config = ASR_MODELS[asr_model_name]
    text_config = TEXT_EMBED_MODELS[text_model_name]
    
    if not st.session_state.asr_model_loaded:
        with st.spinner("Loading ASR models for multi-modal search..."):
            asr_models = load_asr_models(asr_config["model_name"], text_config["model_name"])
            st.session_state.asr_model_loaded = True
    else:
        # Models already loaded, retrieve from cache
        asr_models = load_asr_models(asr_config["model_name"], text_config["model_name"])

# Sidebar for settings  
chunk_duration = st.sidebar.slider("Audio chunk duration (seconds)", 5, 30, 10)
top_k = st.sidebar.slider("Number of results to show", 1, 10, 3)

# Performance metrics in sidebar
st.sidebar.subheader("📊 Performance Stats")

if st.session_state.metrics_log:
    # Get latest metrics for each operation type
    latest_metrics = {}
    for metric in reversed(st.session_state.metrics_log):
        op = metric['operation']
        if op not in latest_metrics:
            latest_metrics[op] = metric
    
    # Model Performance
    if 'Model Loading' in latest_metrics:
        model_metric = latest_metrics['Model Loading']
        with st.sidebar.expander("🤖 Model Performance", expanded=True):
            st.metric("Load Time", f"{model_metric['duration_s']:.1f}s")
            if 'estimated_size_mb' in model_metric['details']:
                st.metric("Model Size", f"{model_metric['details']['estimated_size_mb']:.0f}MB")
            if 'model_params' in model_metric['details']:
                params = model_metric['details']['model_params']
                st.write(f"**Parameters:** {params}")
    
    # Database Stats
    if st.session_state.audio_database:
        with st.sidebar.expander("💾 Database Stats", expanded=True):
            total_clips = len(st.session_state.audio_database)
            total_size_kb = sum(item['embedding'].nbytes for item in st.session_state.audio_database) / 1024
            total_duration = sum(item['end_time'] - item['start_time'] for item in st.session_state.audio_database)
            
            st.metric("Total Clips", total_clips)
            st.metric("Storage", f"{total_size_kb:.1f}KB")
            st.metric("Audio Duration", f"{total_duration:.1f}s")
    
    # Recent Operations
    with st.sidebar.expander("⚡ Recent Operations"):
        recent_ops = list(reversed(st.session_state.metrics_log))[:3]
        for metric in recent_ops:
            st.write(f"**{metric['operation']}**: {metric['duration_ms']:.0f}ms")
            st.caption(f"⏰ {metric['timestamp']}")
    
    # Search Performance (if any searches performed)
    search_metrics = [m for m in st.session_state.metrics_log if m['operation'] == 'Cross-Modal Retrieval']
    if search_metrics:
        latest_search = search_metrics[-1]
        with st.sidebar.expander("🔍 Search Performance"):
            st.metric("Last Search", f"{latest_search['duration_ms']:.0f}ms")
            if 'database_size' in latest_search['details']:
                st.write(f"**DB Size:** {latest_search['details']['database_size']} clips")
            if 'best_similarity_score' in latest_search['details']:
                score = latest_search['details']['best_similarity_score']
                st.write(f"**Best Match:** {score:.3f}")

else:
    st.sidebar.info("📈 Upload files and search to see performance metrics")

# Clear metrics button
if st.sidebar.button("🗑️ Clear Metrics", help="Reset all performance tracking"):
    st.session_state.metrics_log = []
    st.experimental_rerun()

# File upload
st.header("📁 Upload Audio Files")
uploaded_files = st.file_uploader(
    "Choose audio files", 
    type=["mp3", "wav", "m4a", "flac"], 
    accept_multiple_files=True,
    help="Upload multiple audio files to build your searchable database"
)

if uploaded_files:
    st.header("🔍 Processing Files")
    
    # Process uploaded files
    for uploaded_file in uploaded_files:
        # Check if already processed
        if any(item['filename'] == uploaded_file.name for item in st.session_state.audio_database):
            st.info(f"✅ {uploaded_file.name} already processed")
            continue
            
        st.info(f"Processing {uploaded_file.name}...")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        try:
            # Process audio
            chunks, start_times, sr = load_and_process_audio(tmp_path, chunk_duration=chunk_duration)
            
            progress_bar = st.progress(0)
            
            # Get embeddings for each chunk
            for i, (chunk, start_time) in enumerate(zip(chunks, start_times)):
                # Get raw audio embedding
                audio_embedding = get_audio_embedding(chunk, model, processor, sample_rate=sr)
                
                # Get ASR embedding if multi-modal is enabled
                asr_embedding = None
                transcription = None
                if enable_multimodal and asr_models:
                    whisper_model, whisper_processor, text_embedder = asr_models
                    asr_embedding, transcription = get_asr_embedding(
                        chunk, whisper_model, whisper_processor, text_embedder, sample_rate=sr
                    )
                
                # Save chunk for playback
                chunk_filename = f"{uploaded_file.name}_chunk_{i}.wav"
                chunk_path = save_audio_chunk(chunk, sr, chunk_filename)
                
                # Store in database
                item = {
                    'filename': uploaded_file.name,
                    'chunk_id': i,
                    'start_time': start_time,
                    'end_time': start_time + chunk_duration,
                    'embedding': audio_embedding,
                    'audio_path': chunk_path,
                    'chunk_filename': chunk_filename,
                    'transcription': transcription,
                    'asr_embedding': asr_embedding,
                    'is_multimodal': enable_multimodal
                }
                
                st.session_state.audio_database.append(item)
                progress_bar.progress((i + 1) / len(chunks))
            
            st.success(f"✅ Processed {uploaded_file.name} - {len(chunks)} chunks")
            
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        finally:
            # Clean up temp file
            try:
                os.remove(tmp_path)
            except:
                pass

# Display database info
if st.session_state.audio_database:
    st.header("📊 Audio Database")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total clips", len(st.session_state.audio_database))
    with col2:
        files_processed = list(set(item['filename'] for item in st.session_state.audio_database))
        st.metric("Files processed", len(files_processed))
    with col3:
        total_embedding_size = sum(item['embedding'].nbytes for item in st.session_state.audio_database)
        st.metric("Total embeddings size", f"{total_embedding_size / 1024:.1f} KB")
    
    # Show files processed
    st.write(f"**Files:** {', '.join(files_processed)}")
    
    # Database metrics
    if st.checkbox("📋 Show Database Details"):
        total_duration = sum(item['end_time'] - item['start_time'] for item in st.session_state.audio_database)
        avg_embedding_size = np.mean([item['embedding'].nbytes for item in st.session_state.audio_database])
        embedding_dimensions = len(st.session_state.audio_database[0]['embedding']) if st.session_state.audio_database else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total audio duration:** {total_duration:.1f} seconds")
            st.write(f"**Average chunk duration:** {total_duration/len(st.session_state.audio_database):.1f}s")
        with col2:
            st.write(f"**Embedding dimensions:** {embedding_dimensions}")
            st.write(f"**Average embedding size:** {avg_embedding_size/1024:.2f} KB")
    
    # Clear database button
    if st.button("🗑️ Clear Database"):
        st.session_state.audio_database = []
        st.experimental_rerun()

# Search interface
if st.session_state.audio_database:
    st.header("🎯 Search Audio with Text")
    
    st.write("**How it works:** Type what you're looking for (e.g., 'music with drums', 'speech', 'bird sounds', 'guitar solo') and the AI will find matching audio clips.")
    
    # Text query input
    query_text = st.text_input(
        "🔍 Describe the audio you're looking for:",
        placeholder="e.g., 'upbeat music', 'dog barking', 'piano melody', 'speech'",
        help="Use natural language to describe the type of audio you want to find"
    )
    
    if query_text.strip():
        # Search configuration  
        col1, col2 = st.columns([3, 1])
        
        with col1:
            top_k = st.slider("Number of results to show:", 1, 10, 5)
        with col2:
            st.write("") # Spacing
        
        if enable_multimodal:
            st.subheader("🎛️ Multi-Modal Fusion Strategy")
            fusion_strategy = st.selectbox(
                "Choose how to combine audio and speech embeddings:",
                ["Fixed 50/50", "Dynamic Selection", "Adaptive Weighting"],
                help="Different strategies for combining raw audio and speech transcription"
            )
        
        if st.button("🔍 Search Audio Database", type="primary"):
            with st.spinner("Searching for audio matching your description..."):
                try:
        
        with col1:
            if st.button("� Fixed 50/50", help="Equal weighting of audio and ASR", use_container_width=True):
                st.session_state.selected_strategy = "Fixed 50/50"
        
        with col2:
            if st.button("🎯 Dynamic Selection", help="Choose audio OR ASR based on query", use_container_width=True):
                st.session_state.selected_strategy = "Dynamic Selection"
        
        with col3:
            if st.button("🧠 Adaptive Weights", help="Smart weighting based on query analysis", use_container_width=True):
                st.session_state.selected_strategy = "Adaptive Weighting"
        
        with col4:
            if st.button("📊 Compare All", help="Run all strategies and compare results", use_container_width=True):
                st.session_state.selected_strategy = "Compare All"
        
        # Show selected strategy
        if 'selected_strategy' not in st.session_state:
            st.session_state.selected_strategy = fusion_strategy if enable_multimodal else "Audio Only"
        
        current_strategy = st.session_state.selected_strategy
        st.info(f"**Selected Strategy:** {current_strategy}")
        
        if st.button("�🔍 Search Audio Database", type="primary", use_container_width=True):
            if current_strategy == "Compare All" and enable_multimodal:
                # Run all strategies and compare
                st.subheader("📊 Strategy Comparison Results")
                
                all_results = {}
                all_fusion_info = {}
                
                # Test each strategy
                strategies_to_test = ["Fixed 50/50", "Dynamic Selection", "Adaptive Weighting"]
                if not enable_multimodal:
                    strategies_to_test = ["Audio Only"]
                
                for strategy in strategies_to_test:
                    with st.spinner(f"Running {strategy} strategy..."):
                        similarities, fusion_info, retrieval_time = run_search_strategy(
                            query_text, strategy, model, processor, asr_models, 
                            st.session_state.audio_database, enable_multimodal
                        )
                        all_results[strategy] = similarities
                        all_fusion_info[strategy] = fusion_info
                
                # Display comparison
                display_comparison_results(query_text, all_results, all_fusion_info, top_k)
                
            else:
                # Run single strategy
                with st.spinner("Searching for audio matching your description..."):
                    similarities, fusion_info, retrieval_time = run_search_strategy(
                        query_text, current_strategy, model, processor, asr_models,
                        st.session_state.audio_database, enable_multimodal
                    )
                    
                    display_single_strategy_results(query_text, similarities, fusion_info, top_k, current_strategy)
                
                try:
                    # Get text embedding using CLAP's text encoder
                    query_start = time.time()
                    query_embedding = get_text_embedding(query_text, model, processor)
                    query_time = time.time() - query_start
                    
                    # Perform search based on mode
                    similarity_start = time.time()
                    similarities = []
                    fusion_info = ""
                    
                    # Check if database has multi-modal data
                    has_multimodal = any(item.get('is_multimodal', False) for item in st.session_state.audio_database)
                    
                    if enable_multimodal and has_multimodal and asr_models:
                        # Multi-modal search with different fusion strategies
                        
                        if fusion_strategy == "Fixed 50/50":
                            # Equal weighting of audio and ASR embeddings
                            fusion_info = "Using 50/50 fixed weighting (Audio: 50%, ASR: 50%)"
                            
                            for item in st.session_state.audio_database:
                                # Audio similarity
                                audio_sim = cosine_similarity([query_embedding], [item['embedding']])[0][0]
                                
                                # ASR similarity (if available)
                                if item.get('asr_embedding') is not None:
                                    asr_sim = cosine_similarity([query_embedding], [item['asr_embedding']])[0][0]
                                    # Fixed 50/50 combination
                                    combined_sim = 0.5 * audio_sim + 0.5 * asr_sim
                                else:
                                    # Fallback to audio only if no ASR
                                    combined_sim = audio_sim
                                
                                similarities.append((combined_sim, item))
                        
                        elif fusion_strategy == "Dynamic Selection":
                            # Choose either audio OR ASR based on query analysis
                            is_speech, confidence, reasoning = analyze_query_type(query_text)
                            
                            if is_speech:
                                fusion_info = f"Speech-focused query detected (conf: {confidence:.2f}) → Using ASR embeddings only"
                                # Use ASR embeddings only
                                for item in st.session_state.audio_database:
                                    if item.get('asr_embedding') is not None:
                                        sim = cosine_similarity([query_embedding], [item['asr_embedding']])[0][0]
                                    else:
                                        # Fallback to audio if no ASR
                                        sim = cosine_similarity([query_embedding], [item['embedding']])[0][0]
                                    similarities.append((sim, item))
                            else:
                                fusion_info = f"Audio-focused query detected (conf: {confidence:.2f}) → Using raw audio embeddings only"
                                # Use audio embeddings only
                                for item in st.session_state.audio_database:
                                    sim = cosine_similarity([query_embedding], [item['embedding']])[0][0]
                                    similarities.append((sim, item))
                            
                            fusion_info += f"\nReasoning: {reasoning}"
                        
                        elif fusion_strategy == "Adaptive Weighting":
                            # Dynamic weighting based on query analysis
                            audio_weight, asr_weight, explanation = generate_adaptive_weights(query_text)
                            fusion_info = f"Adaptive weighting: {explanation}"
                            
                            for item in st.session_state.audio_database:
                                # Audio similarity
                                audio_sim = cosine_similarity([query_embedding], [item['embedding']])[0][0]
                                
                                # ASR similarity (if available)
                                if item.get('asr_embedding') is not None:
                                    asr_sim = cosine_similarity([query_embedding], [item['asr_embedding']])[0][0]
                                    # Adaptive combination
                                    combined_sim = audio_weight * audio_sim + asr_weight * asr_sim
                                else:
                                    # Fallback to audio only if no ASR
                                    combined_sim = audio_sim
                                
                                similarities.append((combined_sim, item))
                    
                    else:
                        # Standard audio-only search
                        fusion_info = "Audio-only search (multi-modal disabled or no ASR data)"
                        for item in st.session_state.audio_database:
                            sim = cosine_similarity([query_embedding], [item['embedding']])[0][0]
                            similarities.append((sim, item))
                    
                    # Sort by similarity
                    similarities.sort(key=lambda x: x[0], reverse=True)
                    similarity_time = time.time() - similarity_start
                    
                    # Log retrieval metrics
                    retrieval_details = {
                        "query_length": len(query_text),
                        "database_size": len(st.session_state.audio_database),
                        "query_embedding_time_s": round(query_time, 3),
                        "similarity_computation_time_s": round(similarity_time, 3),
                        "total_embeddings_searched": len(similarities),
                        "best_similarity_score": round(similarities[0][0], 3) if similarities else 0,
                        "fusion_strategy": fusion_strategy if enable_multimodal else "Audio-only",
                        "multimodal_enabled": enable_multimodal
                    }
                    
                    total_retrieval_time = query_time + similarity_time
                    log_metric("Cross-Modal Retrieval", total_retrieval_time, retrieval_details)
                    
                    # Display results
                    st.subheader(f"🏆 Top {min(top_k, len(similarities))} Results for: \"{query_text}\"")
                    
                    # Show fusion strategy info
                    if fusion_info:
                        st.info(f"🎭 **Fusion Strategy:** {fusion_info}")
                    
                    if similarities[0][0] < 0.1:  # Very low similarity threshold
                        st.warning("⚠️ Low similarity scores detected. The search didn't find very relevant matches. Try different keywords or add more diverse audio files.")
                    
                    for rank, (similarity, item) in enumerate(similarities[:top_k], 1):
                        with st.container():
                            col1, col2, col3 = st.columns([1, 2, 1])
                            
                            with col1:
                                # Color-code similarity scores
                                if similarity > 0.3:
                                    score_color = "🟢"  # Green for good matches
                                elif similarity > 0.15:
                                    score_color = "🟡"  # Yellow for ok matches  
                                else:
                                    score_color = "🔴"  # Red for poor matches
                                    
                                st.metric(
                                    label=f"#{rank} {score_color}",
                                    value=f"{similarity:.3f}",
                                    help="Similarity score: >0.3=Good, 0.15-0.3=OK, <0.15=Poor"
                                )
                            
                            with col2:
                                st.write(f"**File:** {item['filename']}")
                                st.write(f"**Time:** {item['start_time']:.1f}s - {item['end_time']:.1f}s")
                                
                                # Show transcription if available
                                if item.get('transcription') and item['transcription'] != "[ASR_FAILED]":
                                    with st.expander("📝 Transcription"):
                                        st.write(item['transcription'])
                                
                                # Play audio if file exists
                                if os.path.exists(item['audio_path']):
                                    with open(item['audio_path'], 'rb') as audio_file:
                                        audio_bytes = audio_file.read()
                                        st.audio(audio_bytes, format='audio/wav')
                                else:
                                    st.warning("Audio file not found")
                            
                            with col3:
                                # Show embedding info
                                st.write("**Embedding Info:**")
                                if item.get('is_multimodal'):
                                    st.write("🎭 Multi-modal")
                                    if item.get('transcription'):
                                        st.write(f"📝 ASR: ✅")
                                    else:
                                        st.write(f"📝 ASR: ❌")
                                else:
                                    st.write("🎵 Audio-only")
                            
                            st.divider()
                    
                    # Add some example queries
                    st.markdown("---")
                    st.markdown("**💡 Example queries to try:**")
                    
                    if enable_multimodal:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**🎵 Audio-focused queries:**")
                            audio_queries = [
                                "music with drums", "guitar solo", "piano music", 
                                "electronic beats", "nature sounds", "bird sounds"
                            ]
                            st.write(" • ".join([f"`{q}`" for q in audio_queries]))
                        
                        with col2:
                            st.markdown("**📝 Speech-focused queries:**")
                            speech_queries = [
                                "speech or talking", "someone speaking", "conversation",
                                "person saying hello", "interview", "narrator voice"
                            ]
                            st.write(" • ".join([f"`{q}`" for q in speech_queries]))
                    else:
                        example_queries = [
                            "music with drums", "speech or talking", "bird sounds", 
                            "guitar solo", "piano music", "electronic beats",
                            "nature sounds", "instrumental music", "vocal singing"
                        ]
                        st.write(" • ".join([f"`{q}`" for q in example_queries]))
                
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
                    st.error("This might be due to model loading issues. Try refreshing the page.")

else:
    st.info("👆 Upload some audio files above to start building your searchable database!")

# ============================================================================
# COMPREHENSIVE METRICS DASHBOARD
# ============================================================================

if st.checkbox("📊 Advanced Metrics Dashboard"):
    st.header("📈 Performance Analytics")
    
    if st.session_state.metrics_log:
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_operations = len(st.session_state.metrics_log)
            st.metric("Total Operations", total_operations)
        
        with col2:
            avg_time = np.mean([m['duration_s'] for m in st.session_state.metrics_log])
            st.metric("Avg Operation Time", f"{avg_time:.3f}s")
        
        with col3:
            model_loads = [m for m in st.session_state.metrics_log if m['operation'] == 'Model Loading']
            st.metric("Model Loads", len(model_loads))
        
        with col4:
            searches = [m for m in st.session_state.metrics_log if m['operation'] == 'Cross-Modal Retrieval']
            st.metric("Searches Performed", len(searches))
        
        # Operation breakdown
        st.subheader("⏱️ Operation Timing Breakdown")
        
        operation_times = {}
        for metric in st.session_state.metrics_log:
            op = metric['operation']
            if op not in operation_times:
                operation_times[op] = []
            operation_times[op].append(metric['duration_s'])
        
        for operation, times in operation_times.items():
            avg_time = np.mean(times)
            total_time = np.sum(times)
            count = len(times)
            
            st.write(f"**{operation}**: {count} times, avg: {avg_time:.3f}s, total: {total_time:.3f}s")
        
        # Detailed metrics table
        st.subheader("🔍 Detailed Operation Log")
        
        if st.button("📥 Download Metrics as CSV"):
            import pandas as pd
            df = pd.DataFrame(st.session_state.metrics_log)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"audio_search_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Show recent operations
        recent_count = st.slider("Show last N operations", 5, len(st.session_state.metrics_log), min(10, len(st.session_state.metrics_log)))
        recent_metrics = st.session_state.metrics_log[-recent_count:]
        
        for i, metric in enumerate(reversed(recent_metrics)):
            with st.expander(f"{metric['timestamp']} - {metric['operation']} ({metric['duration_ms']}ms)"):
                st.json(metric)
    
    else:
        st.info("No metrics collected yet. Upload some files and perform searches to see performance data!")

# ============================================================================
# MODEL COMPARISON SECTION  
# ============================================================================

if st.checkbox("🔬 Model Comparison Mode"):
    st.header("🆚 Compare Different Models")
    st.info("⚠️ This will reload models and clear the database. Use this to test different embedding models on the same audio files.")
    
    st.write("**Available Models:**")
    for name, config in AVAILABLE_MODELS.items():
        current = "✅ Currently loaded" if st.session_state.current_model == config["model_name"] else ""
        st.write(f"- **{name}**: {config['description']} {current}")
    
    if st.button("🔄 Reset for Model Testing"):
        st.session_state.audio_database = []
        st.session_state.metrics_log = []
        st.cache_resource.clear()
        st.success("Reset complete! Now try different models and compare their performance.")
        st.experimental_rerun()

# Footer
st.markdown("---")
if enable_multimodal:
    st.markdown("""
    **🎭 Multi-Modal Search:** This app combines two complementary approaches:
    - **Raw Audio Embeddings (CLAP)**: Captures acoustic properties, music, sound effects
    - **ASR + Text Embeddings**: Captures speech content and semantic meaning
    
    **Fusion Strategies:**
    - **Fixed 50/50**: Equal weighting of both modalities
    - **Dynamic Selection**: Choose audio OR speech embeddings based on query type
    - **Adaptive Weighting**: Dynamically balance modalities based on query analysis
    """)
else:
    st.markdown("**How it works:** This app uses CLAP (Contrastive Language-Audio Pre-training) to create aligned embeddings for both text and audio. When you search with text, it finds audio clips that semantically match your description, even without any speech or transcription.")