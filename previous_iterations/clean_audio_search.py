"""
Clean Audio Search Demo - Completely Fixed Version
Multi-modal audio search with ASR and audio captioning
"""

import streamlit as st
import torch
import librosa
import numpy as np
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from sentence_transformers import SentenceTransformer
import tempfile
import os
from typing import List, Dict
import time
from dataclasses import dataclass

@dataclass
class ModelStats:
    """Simple model performance tracking"""
    model_name: str
    load_time: float = 0.0
    total_calls: int = 0
    avg_processing_time: float = 0.0
    success_rate: float = 1.0

class UnifiedAudioSearch:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model options
        self.text_embedder_options = {
            "all-MiniLM-L6-v2": "General (384D, fast)",
            "all-mpnet-base-v2": "Better quality (768D)",
            "sentence-transformers/clip-ViT-B-32-multilingual-v1": "Multimodal (512D)",
        }
        
        self.caption_model_options = {
            "cahya/whisper-tiny-audio-captioning-v2.0": "Cahya Tiny v2.0",
            "MU-NLPC/whisper-tiny-audio-captioning": "MU-NLPC AudioSet",
        }
        
        self.asr_model_options = {
            "openai/whisper-tiny": "Whisper Tiny (39MB)",
            "openai/whisper-base": "Whisper Base (74MB)",
            "openai/whisper-small": "Whisper Small (244MB)"
        }
        
        # Current selections
        self.text_embedder_name = "all-MiniLM-L6-v2"
        self.caption_model_name = "cahya/whisper-tiny-audio-captioning-v2.0"
        self.asr_model_name = "openai/whisper-tiny"
        
        # Models (loaded on demand)
        self.text_embedder = None
        self.asr_pipe = None  # Using pipeline for ASR
        self.caption_processor = None
        self.caption_model = None
        
        # Statistics
        self.stats = {
            'text_embedder': ModelStats("none"),
            'asr_model': ModelStats("none"),
            'caption_model': ModelStats("none")
        }
        
        # Audio database
        self.audio_database = []
        
    def load_models(self):
        """Load all models with proper error handling"""
        
        # 1. Text Embedder
        if (self.text_embedder is None or 
            self.stats['text_embedder'].model_name != self.text_embedder_name):
            
            with st.spinner(f"Loading text embedder: {self.text_embedder_name}..."):
                start_time = time.time()
                try:
                    self.text_embedder = SentenceTransformer(self.text_embedder_name)
                    load_time = time.time() - start_time
                    self.stats['text_embedder'] = ModelStats(self.text_embedder_name, load_time)
                    st.success(f"✅ Text embedder loaded ({load_time:.1f}s)")
                except Exception as e:
                    st.error(f"❌ Failed to load text embedder: {e}")
                    return False
        
        # 2. ASR Pipeline  
        if (self.asr_pipe is None or 
            self.stats['asr_model'].model_name != self.asr_model_name):
            
            with st.spinner(f"Loading ASR model: {self.asr_model_name}..."):
                start_time = time.time()
                try:
                    self.asr_pipe = pipeline(
                        "automatic-speech-recognition",
                        model=self.asr_model_name,
                        device=0 if torch.cuda.is_available() else -1,
                        return_timestamps=False
                    )
                    load_time = time.time() - start_time
                    self.stats['asr_model'] = ModelStats(self.asr_model_name, load_time)
                    st.success(f"✅ ASR model loaded ({load_time:.1f}s)")
                except Exception as e:
                    st.error(f"❌ Failed to load ASR model: {e}")
                    return False
        
        # 3. Caption Model
        if (self.caption_model is None or 
            self.stats['caption_model'].model_name != self.caption_model_name):
            
            with st.spinner(f"Loading caption model: {self.caption_model_name}..."):
                start_time = time.time()
                try:
                    self.caption_processor = WhisperProcessor.from_pretrained(self.caption_model_name)
                    self.caption_model = WhisperForConditionalGeneration.from_pretrained(self.caption_model_name)
                    self.caption_model.to(self.device)
                    load_time = time.time() - start_time
                    self.stats['caption_model'] = ModelStats(self.caption_model_name, load_time)
                    st.success(f"✅ Caption model loaded ({load_time:.1f}s)")
                except Exception as e:
                    st.error(f"❌ Failed to load caption model: {e}")
                    return False
        
        return True
    
    def process_audio_file(self, audio_file) -> List[Dict]:
        """Process audio file into searchable segments"""
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Load and normalize audio
            audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
            audio = librosa.util.normalize(audio)
            
            # Create 5-second segments
            segment_length = 5 * sr
            segments = []
            
            for i in range(0, len(audio), segment_length):
                segment_audio = audio[i:i + segment_length]
                
                # Skip short segments
                if len(segment_audio) < sr:
                    continue
                
                start_time = i / sr
                end_time = min((i + segment_length) / sr, len(audio) / sr)
                
                # Get ASR and caption
                asr_text = self._get_asr_transcription(segment_audio)
                caption_text = self._get_audio_caption(segment_audio, sr)
                
                # Create embeddings
                asr_embedding = None
                caption_embedding = None
                combined_text = ""
                
                if asr_text.strip():
                    asr_embedding = self.text_embedder.encode(asr_text)
                    combined_text += asr_text + " "
                
                if caption_text.strip():
                    caption_embedding = self.text_embedder.encode(caption_text)
                    combined_text += caption_text
                
                # Only store if we have some text
                if combined_text.strip():
                    combined_embedding = self.text_embedder.encode(combined_text.strip())
                    
                    segments.append({
                        'segment_id': f"seg_{len(segments)}",
                        'start_time': start_time,
                        'end_time': end_time,
                        'asr_text': asr_text,
                        'caption_text': caption_text,
                        'combined_text': combined_text.strip(),
                        'asr_embedding': asr_embedding,
                        'caption_embedding': caption_embedding,
                        'combined_embedding': combined_embedding,
                        'audio_data': segment_audio,
                        'sample_rate': sr
                    })
            
            return segments
            
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def _get_asr_transcription(self, audio: np.ndarray) -> str:
        """Get ASR transcription using pipeline"""
        start_time = time.time()
        
        try:
            # Use pipeline with clean parameters
            result = self.asr_pipe(
                audio,
                generate_kwargs={
                    "language": "en",
                    "task": "transcribe",
                    "temperature": 0.0,
                    "no_repeat_ngram_size": 2
                }
            )
            
            text = result["text"].strip() if isinstance(result, dict) else str(result).strip()
            
            # Simple filtering
            if (text and len(text) > 3 and 
                not text.lower().startswith(('you', 'yeah', 'mm-hmm', 'uh', 'um')) and
                'laionionion' not in text.lower()):
                
                # Update stats
                processing_time = time.time() - start_time
                self._update_stats('asr_model', processing_time, success=True)
                return text
            else:
                self._update_stats('asr_model', time.time() - start_time, success=False)
                return ""
                
        except Exception as e:
            print(f"ASR Error: {e}")
            self._update_stats('asr_model', time.time() - start_time, success=False)
            return ""
    
    def _get_audio_caption(self, audio: np.ndarray, sr: int) -> str:
        """Get audio caption"""
        start_time = time.time()
        
        try:
            # Process audio
            input_features = self.caption_processor(
                audio, 
                sampling_rate=sr, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate caption
            with torch.no_grad():
                predicted_ids = self.caption_model.generate(
                    input_features,
                    max_length=80,
                    no_repeat_ngram_size=3,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.2
                )
                
                caption = self.caption_processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0].strip()
            
            # Filter garbage
            if (caption and len(caption) > 5 and 
                'laionionion' not in caption.lower() and
                not caption.lower().startswith(('you', 'yeah'))):
                
                processing_time = time.time() - start_time
                self._update_stats('caption_model', processing_time, success=True)
                return caption
            else:
                self._update_stats('caption_model', time.time() - start_time, success=False)
                return ""
                
        except Exception as e:
            print(f"Caption Error: {e}")
            self._update_stats('caption_model', time.time() - start_time, success=False)
            return ""
    
    def _update_stats(self, model_key: str, processing_time: float, success: bool = True):
        """Update model statistics"""
        stats = self.stats[model_key]
        stats.total_calls += 1
        
        # Update average processing time
        stats.avg_processing_time = (
            (stats.avg_processing_time * (stats.total_calls - 1) + processing_time) 
            / stats.total_calls
        )
        
        # Update success rate
        stats.success_rate = (
            (stats.success_rate * (stats.total_calls - 1) + (1.0 if success else 0.0)) 
            / stats.total_calls
        )
    
    def search_audio(self, query: str, search_mode: str = "combined") -> List[Dict]:
        """Search audio database"""
        if not self.audio_database:
            return []
        
        # Create query embedding
        query_embedding = self.text_embedder.encode(query)
        
        results = []
        for segment in self.audio_database:
            similarity = 0.0
            
            if search_mode == "combined" and segment.get('combined_embedding') is not None:
                similarity = float(np.dot(query_embedding, segment['combined_embedding']))
            elif search_mode == "asr" and segment.get('asr_embedding') is not None:
                similarity = float(np.dot(query_embedding, segment['asr_embedding']))
            elif search_mode == "caption" and segment.get('caption_embedding') is not None:
                similarity = float(np.dot(query_embedding, segment['caption_embedding']))
            
            if similarity > 0.1:  # Threshold
                results.append({
                    **segment,
                    'similarity': similarity
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:10]  # Top 10 results

# Streamlit App
def main():
    st.set_page_config(page_title="Audio Search - Fixed", layout="wide")
    st.title("🎵 Multi-Modal Audio Search (Fixed)")
    
    # Initialize search system
    if 'search_system' not in st.session_state:
        st.session_state.search_system = UnifiedAudioSearch()
    
    search_system = st.session_state.search_system
    
    # Sidebar - Model Configuration
    with st.sidebar:
        st.header("🔧 Model Configuration")
        
        # Text Embedder
        new_embedder = st.selectbox(
            "Text Embedder",
            options=list(search_system.text_embedder_options.keys()),
            index=list(search_system.text_embedder_options.keys()).index(search_system.text_embedder_name),
            format_func=lambda x: f"{x} - {search_system.text_embedder_options[x]}"
        )
        
        # ASR Model
        new_asr = st.selectbox(
            "ASR Model",
            options=list(search_system.asr_model_options.keys()),
            index=list(search_system.asr_model_options.keys()).index(search_system.asr_model_name),
            format_func=lambda x: f"{x} - {search_system.asr_model_options[x]}"
        )
        
        # Caption Model
        new_caption = st.selectbox(
            "Caption Model", 
            options=list(search_system.caption_model_options.keys()),
            index=list(search_system.caption_model_options.keys()).index(search_system.caption_model_name),
            format_func=lambda x: f"{x} - {search_system.caption_model_options[x]}"
        )
        
        # Update models if changed
        if (new_embedder != search_system.text_embedder_name or
            new_asr != search_system.asr_model_name or
            new_caption != search_system.caption_model_name):
            
            search_system.text_embedder_name = new_embedder
            search_system.asr_model_name = new_asr
            search_system.caption_model_name = new_caption
            
            # Reset models to force reload
            search_system.text_embedder = None
            search_system.asr_pipe = None
            search_system.caption_model = None
        
        st.divider()
        
        # Model Statistics
        st.header("📊 Statistics")
        for model_name, stats in search_system.stats.items():
            if stats.model_name != "none":
                st.metric(
                    f"{model_name.replace('_', ' ').title()}",
                    f"{stats.total_calls} calls",
                    f"{stats.avg_processing_time:.3f}s avg, {stats.success_rate:.1%} success"
                )
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Audio")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Upload audio for processing and search"
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file)
            
            if st.button("🔄 Process Audio", type="primary"):
                # Load models first
                if search_system.load_models():
                    with st.spinner("Processing audio..."):
                        try:
                            segments = search_system.process_audio_file(uploaded_file)
                            search_system.audio_database.extend(segments)
                            
                            st.success(f"✅ Processed {len(segments)} segments")
                            st.json({
                                "total_segments": len(search_system.audio_database),
                                "new_segments": len(segments)
                            })
                            
                        except Exception as e:
                            st.error(f"❌ Processing failed: {e}")
    
    with col2:
        st.header("🔍 Search Audio")
        
        if search_system.audio_database:
            query = st.text_input(
                "Search query",
                placeholder="e.g., 'music with drums', 'person speaking', 'upbeat song'"
            )
            
            search_mode = st.selectbox(
                "Search Mode",
                ["combined", "asr", "caption"],
                format_func=lambda x: {
                    "combined": "🔄 Combined (ASR + Caption)",
                    "asr": "🎤 ASR Only (Speech)",
                    "caption": "🎵 Caption Only (Audio Description)"
                }[x]
            )
            
            if query and st.button("🔎 Search", type="primary"):
                with st.spinner("Searching..."):
                    results = search_system.search_audio(query, search_mode)
                    
                    if results:
                        st.subheader(f"🎯 Found {len(results)} matches")
                        
                        for i, result in enumerate(results):
                            with st.expander(f"Result {i+1} (similarity: {result['similarity']:.3f})"):
                                st.write(f"**Time:** {result['start_time']:.1f}s - {result['end_time']:.1f}s")
                                if result.get('asr_text'):
                                    st.write(f"**Speech:** {result['asr_text']}")
                                if result.get('caption_text'):
                                    st.write(f"**Description:** {result['caption_text']}")
                                
                                # Play audio segment
                                st.audio(result['audio_data'], sample_rate=result['sample_rate'])
                    else:
                        st.info("No matches found. Try different keywords.")
        else:
            st.info("👆 Upload and process audio files first to enable search")

if __name__ == "__main__":
    main()