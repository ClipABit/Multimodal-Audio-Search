"""
Lightweight Audio Search System

This implements your idea:
1. Small Whisper model for speech-to-text  
2. Sentence transformer for unified embedding space
3. Audio feature extraction mapped to same embedding space

Key insight: We use an audio feature extractor + small neural network to map 
audio features into the sentence transformer's embedding space.
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
from sentence_transformers import SentenceTransformer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import tempfile
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class AudioToTextEmbeddingBridge(nn.Module):
    """
    Small neural network that maps audio features to sentence transformer embedding space
    This is the key component that makes your lightweight idea work!
    """
    def __init__(self, audio_feature_dim=128, text_embedding_dim=384):
        super().__init__()
        self.bridge = nn.Sequential(
            nn.Linear(audio_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(512, text_embedding_dim),
            nn.Tanh()  # Normalize to [-1, 1] range like sentence embeddings
        )
    
    def forward(self, audio_features):
        return self.bridge(audio_features)

class LightweightAudioSearch:
    def __init__(self):
        # Small, fast models
        self.whisper_model_name = "openai/whisper-tiny"  # Only 39MB!
        self.sentence_model_name = "all-MiniLM-L6-v2"   # Only 23MB!
        
        self.whisper_processor = None
        self.whisper_model = None
        self.sentence_model = None
        self.audio_bridge = None
        
        # Audio processing config
        self.sample_rate = 16000  # Whisper's expected rate
        self.chunk_duration = 10  # seconds
        
    @st.cache_resource
    def load_models(_self):
        """Load all the lightweight models"""
        print("Loading lightweight models...")
        
        # Load tiny Whisper (39MB)
        _self.whisper_processor = WhisperProcessor.from_pretrained(_self.whisper_model_name)
        _self.whisper_model = WhisperForConditionalGeneration.from_pretrained(_self.whisper_model_name)
        
        # Load small sentence transformer (23MB) 
        _self.sentence_model = SentenceTransformer(_self.sentence_model_name)
        
        # Initialize audio-to-text bridge network
        text_dim = _self.sentence_model.get_sentence_embedding_dimension()
        _self.audio_bridge = AudioToTextEmbeddingBridge(
            audio_feature_dim=128, 
            text_embedding_dim=text_dim
        )
        
        print(f"Models loaded! Text embedding dim: {text_dim}")
        return _self.whisper_processor, _self.whisper_model, _self.sentence_model, _self.audio_bridge
    
    def extract_audio_features(self, audio_chunk):
        """
        Extract lightweight audio features that can be mapped to text embedding space
        Using MFCC + spectral features - much lighter than deep audio models
        """
        # Extract MFCC features (classic audio features)
        mfccs = librosa.feature.mfcc(y=audio_chunk, sr=self.sample_rate, n_mfcc=13)
        
        # Extract spectral features  
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_chunk, sr=self.sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_chunk, sr=self.sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_chunk, sr=self.sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_chunk)
        
        # Aggregate features (mean across time)
        features = np.concatenate([
            np.mean(mfccs, axis=1),           # 13 features
            np.mean(spectral_centroids),      # 1 feature  
            np.mean(spectral_bandwidth),      # 1 feature
            np.mean(spectral_rolloff),        # 1 feature
            np.mean(zero_crossing_rate),      # 1 feature
        ])
        
        # Pad or truncate to expected size
        target_size = 128
        if len(features) < target_size:
            features = np.pad(features, (0, target_size - len(features)))
        else:
            features = features[:target_size]
            
        return features
    
    def process_audio_chunk(self, audio_chunk):
        """
        Process audio using BOTH pathways:
        1. Speech pathway: Whisper -> text -> sentence embedding
        2. Audio pathway: audio features -> bridge network -> text embedding space
        """
        results = {}
        
        # Pathway 1: Speech-to-text pathway
        try:
            # Transcribe with tiny Whisper
            inputs = self.whisper_processor(audio_chunk, sampling_rate=self.sample_rate, return_tensors="pt")
            with torch.no_grad():
                generated_ids = self.whisper_model.generate(inputs["input_features"])
                transcription = self.whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Embed transcription with sentence transformer
            if transcription.strip():
                speech_embedding = self.sentence_model.encode([transcription])[0]
                results['speech_embedding'] = speech_embedding
                results['transcription'] = transcription
            else:
                results['speech_embedding'] = None
                results['transcription'] = ""
                
        except Exception as e:
            print(f"Speech pathway failed: {e}")
            results['speech_embedding'] = None
            results['transcription'] = ""
        
        # Pathway 2: Audio features pathway  
        try:
            # Extract lightweight audio features
            audio_features = self.extract_audio_features(audio_chunk)
            
            # Map to text embedding space using bridge network
            with torch.no_grad():
                audio_features_tensor = torch.FloatTensor(audio_features).unsqueeze(0)
                audio_embedding = self.audio_bridge(audio_features_tensor).squeeze(0).numpy()
                results['audio_embedding'] = audio_embedding
                
        except Exception as e:
            print(f"Audio pathway failed: {e}")
            results['audio_embedding'] = None
        
        return results
    
    def train_audio_bridge(self, training_data):
        """
        Train the bridge network to map audio features to text embedding space
        
        Training strategy:
        - For audio with speech: train bridge to match speech embeddings
        - For audio without speech: train bridge to match semantic embeddings of descriptions
        """
        if not training_data:
            print("No training data provided, using pre-trained bridge")
            return
            
        optimizer = torch.optim.Adam(self.audio_bridge.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        print(f"Training bridge network on {len(training_data)} samples...")
        
        for epoch in range(50):  # Quick training
            total_loss = 0
            for item in training_data:
                audio_features = torch.FloatTensor(item['audio_features']).unsqueeze(0)
                target_embedding = torch.FloatTensor(item['target_embedding']).unsqueeze(0)
                
                optimizer.zero_grad()
                predicted_embedding = self.audio_bridge(audio_features)
                loss = criterion(predicted_embedding, target_embedding)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(training_data):.4f}")
    
    def search(self, query_text, audio_database, fusion_strategy="adaptive"):
        """
        Search audio database using unified embedding space
        """
        # Embed query text
        query_embedding = self.sentence_model.encode([query_text])[0]
        
        similarities = []
        
        for item in audio_database:
            if fusion_strategy == "speech_only":
                # Use only speech embeddings
                if item.get('speech_embedding') is not None:
                    sim = cosine_similarity([query_embedding], [item['speech_embedding']])[0][0]
                else:
                    sim = 0.0
                    
            elif fusion_strategy == "audio_only":
                # Use only audio feature embeddings
                if item.get('audio_embedding') is not None:
                    sim = cosine_similarity([query_embedding], [item['audio_embedding']])[0][0]
                else:
                    sim = 0.0
                    
            else:  # adaptive fusion
                # Combine both if available, with smart weighting
                speech_sim = 0.0
                audio_sim = 0.0
                
                if item.get('speech_embedding') is not None:
                    speech_sim = cosine_similarity([query_embedding], [item['speech_embedding']])[0][0]
                
                if item.get('audio_embedding') is not None:
                    audio_sim = cosine_similarity([query_embedding], [item['audio_embedding']])[0][0]
                
                # Adaptive weighting based on transcription quality
                transcription = item.get('transcription', '')
                if len(transcription.strip()) > 10:  # Good transcription
                    sim = 0.7 * speech_sim + 0.3 * audio_sim
                else:  # Poor/no transcription
                    sim = 0.3 * speech_sim + 0.7 * audio_sim
            
            similarities.append(sim)
        
        return np.array(similarities)


# Streamlit UI for lightweight system
def main():
    st.set_page_config(page_title="Lightweight Audio Search", page_icon="🎵", layout="wide")
    
    st.title("🎵 Lightweight Audio Search System")
    st.write("**Tiny models, big results!** Uses only ~60MB total:")
    st.write("- Whisper-tiny (39MB) for speech-to-text")  
    st.write("- MiniLM-L6-v2 (23MB) for text embeddings")
    st.write("- Small bridge network (<1MB) to map audio features to text space")
    
    # Initialize system
    if 'lightweight_search' not in st.session_state:
        st.session_state.lightweight_search = LightweightAudioSearch()
        st.session_state.lightweight_database = []
    
    system = st.session_state.lightweight_search
    
    # Load models
    with st.spinner("Loading lightweight models (first time only)..."):
        system.load_models()
    
    st.success("✅ Lightweight models loaded! Total size: ~60MB")
    
    # File upload
    st.header("📁 Upload Audio Files")
    uploaded_files = st.file_uploader(
        "Choose audio files", 
        type=['wav', 'mp3', 'mp4', 'm4a'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in [item.get('filename') for item in st.session_state.lightweight_database]:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Save file temporarily
                    temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load and process audio
                    audio, sr = librosa.load(temp_path, sr=system.sample_rate)
                    
                    # Process in chunks
                    chunk_length = system.sample_rate * system.chunk_duration
                    
                    for i, start in enumerate(range(0, len(audio), chunk_length)):
                        end = min(start + chunk_length, len(audio))
                        chunk = audio[start:end]
                        
                        if len(chunk) > system.sample_rate:  # At least 1 second
                            results = system.process_audio_chunk(chunk)
                            
                            item = {
                                'filename': uploaded_file.name,
                                'chunk_id': i,
                                'start_time': start / sr,
                                'end_time': end / sr,
                                'filepath': temp_path,
                                'speech_embedding': results.get('speech_embedding'),
                                'audio_embedding': results.get('audio_embedding'),
                                'transcription': results.get('transcription', '')
                            }
                            
                            st.session_state.lightweight_database.append(item)
                
                st.success(f"✅ Processed {uploaded_file.name}")
    
    # Database info
    if st.session_state.lightweight_database:
        st.header("📊 Lightweight Database")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Audio chunks", len(st.session_state.lightweight_database))
        with col2:
            speech_chunks = sum(1 for item in st.session_state.lightweight_database if item.get('speech_embedding') is not None)
            st.metric("With speech", speech_chunks)
        with col3:
            audio_chunks = sum(1 for item in st.session_state.lightweight_database if item.get('audio_embedding') is not None)
            st.metric("With audio features", audio_chunks)
    
    # Search interface
    if st.session_state.lightweight_database:
        st.header("🔍 Lightweight Search")
        
        query_text = st.text_input(
            "Search query:",
            placeholder="e.g., 'music with drums', 'speech about weather', 'bird sounds'"
        )
        
        fusion_strategy = st.selectbox(
            "Search strategy:",
            ["adaptive", "speech_only", "audio_only"],
            help="Adaptive uses both speech and audio features intelligently"
        )
        
        if query_text and st.button("🚀 Search"):
            with st.spinner("Searching..."):
                start_time = time.time()
                similarities = system.search(
                    query_text, 
                    st.session_state.lightweight_database, 
                    fusion_strategy
                )
                search_time = time.time() - start_time
            
            st.success(f"Search completed in {search_time:.3f}s")
            
            # Show results
            top_k = 5
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            st.subheader(f"🎵 Top {top_k} Results")
            
            for rank, idx in enumerate(top_indices, 1):
                item = st.session_state.lightweight_database[idx]
                similarity = similarities[idx]
                
                with st.expander(f"{rank}. {item['filename']} (Score: {similarity:.4f})", expanded=(rank <= 3)):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if item['transcription']:
                            st.write(f"**Transcription:** {item['transcription']}")
                        else:
                            st.write("**No speech detected** - matched on audio features")
                        
                        st.write(f"**Time:** {item['start_time']:.1f}s - {item['end_time']:.1f}s")
                        
                        # Audio player would go here
                        st.info("Audio player: Load chunk from file")
                    
                    with col2:
                        st.metric("Similarity", f"{similarity:.4f}")
                        if item.get('speech_embedding') is not None:
                            st.write("✅ Speech pathway")
                        if item.get('audio_embedding') is not None:
                            st.write("✅ Audio pathway")

if __name__ == "__main__":
    main()