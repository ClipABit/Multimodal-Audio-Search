import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import tempfile
import librosa
import soundfile as sf
from transformers import ClapProcessor, ClapModel
import torch
import psutil
import threading
from datetime import datetime

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

# Session state for metrics
if 'metrics_log' not in st.session_state:
    st.session_state.metrics_log = []
if 'audio_database' not in st.session_state:
    st.session_state.audio_database = []
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

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
selected_model_name = st.sidebar.selectbox(
    "Choose Audio Embedding Model:",
    list(AVAILABLE_MODELS.keys()),
    help="Different models have different strengths. Try them all!"
)

selected_model_config = AVAILABLE_MODELS[selected_model_name]
st.sidebar.write(f"**Description:** {selected_model_config['description']}")

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

# Load model (will reload if user changes model selection)
model_changed = (st.session_state.current_model != selected_model_config["model_name"])
if model_changed:
    st.cache_resource.clear()  # Clear cache to reload with new model

with st.spinner("Loading AI model..." if model_changed else "Model ready"):
    model, processor = load_audio_model(selected_model_config["model_name"])

# Sidebar for settings  
chunk_duration = st.sidebar.slider("Audio chunk duration (seconds)", 5, 30, 10)
top_k = st.sidebar.slider("Number of results to show", 1, 10, 3)

# Performance metrics in sidebar
if st.sidebar.checkbox("📊 Show Performance Metrics", value=False):
    st.sidebar.subheader("📈 Live Metrics")
    
    if st.session_state.metrics_log:
        latest_metrics = st.session_state.metrics_log[-5:]  # Show last 5 operations
        for metric in reversed(latest_metrics):
            st.sidebar.text(f"{metric['timestamp']}: {metric['operation']}")
            st.sidebar.text(f"  ⏱️ {metric['duration_ms']}ms")
            if metric['details']:
                for key, value in metric['details'].items():
                    if isinstance(value, (int, float)) and key.endswith('_s'):
                        st.sidebar.text(f"  📋 {key}: {value}s")
                    else:
                        st.sidebar.text(f"  📋 {key}: {value}")
            st.sidebar.text("---")
    
    if st.sidebar.button("🗑️ Clear Metrics"):
        st.session_state.metrics_log = []

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
                # Get embedding
                embedding = get_audio_embedding(chunk, model, processor, sample_rate=sr)
                
                # Save chunk for playback
                chunk_filename = f"{uploaded_file.name}_chunk_{i}.wav"
                chunk_path = save_audio_chunk(chunk, sr, chunk_filename)
                
                # Store in database
                st.session_state.audio_database.append({
                    'filename': uploaded_file.name,
                    'chunk_id': i,
                    'start_time': start_time,
                    'end_time': start_time + chunk_duration,
                    'embedding': embedding,
                    'audio_path': chunk_path,
                    'chunk_filename': chunk_filename
                })
                
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
        if st.button("🔍 Search Audio Database", type="primary"):
            with st.spinner("Searching for audio matching your description..."):
                
                try:
                    # Get text embedding using CLAP's text encoder
                    query_start = time.time()
                    query_embedding = get_text_embedding(query_text, model, processor)
                    query_time = time.time() - query_start
                    
                    # Calculate similarities with all audio clips
                    similarity_start = time.time()
                    similarities = []
                    for item in st.session_state.audio_database:
                        # Compare text embedding to audio embedding (cross-modal search)
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
                        "best_similarity_score": round(similarities[0][0], 3) if similarities else 0
                    }
                    
                    total_retrieval_time = query_time + similarity_time
                    log_metric("Cross-Modal Retrieval", total_retrieval_time, retrieval_details)
                    
                    # Display results
                    st.subheader(f"🏆 Top {min(top_k, len(similarities))} Results for: \"{query_text}\"")
                    
                    if similarities[0][0] < 0.1:  # Very low similarity threshold
                        st.warning("⚠️ Low similarity scores detected. The search didn't find very relevant matches. Try different keywords or add more diverse audio files.")
                    
                    for rank, (similarity, item) in enumerate(similarities[:top_k], 1):
                        with st.container():
                            col1, col2 = st.columns([1, 3])
                            
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
                                
                                # Play audio if file exists
                                if os.path.exists(item['audio_path']):
                                    with open(item['audio_path'], 'rb') as audio_file:
                                        audio_bytes = audio_file.read()
                                        st.audio(audio_bytes, format='audio/wav')
                                else:
                                    st.warning("Audio file not found")
                            
                            st.divider()
                    
                    # Add some example queries
                    st.markdown("---")
                    st.markdown("**💡 Example queries to try:**")
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
st.markdown("**How it works:** This app uses CLAP (Contrastive Language-Audio Pre-training) to create aligned embeddings for both text and audio. When you search with text, it finds audio clips that semantically match your description, even without any speech or transcription.")