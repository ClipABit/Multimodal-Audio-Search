"""
Dual Pipeline Audio Search System
Two separate working pipelines: ASR + General Audio Analysis
Focus: WORKING over small, with detailed statistics
"""

import streamlit as st
import torch
import librosa
import numpy as np
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from sentence_transformers import SentenceTransformer
import tempfile
import os
import psutil
import platform
import gc
from typing import List, Dict, Tuple
import time
from dataclasses import dataclass, field
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class PipelineStats:
    """Detailed statistics for each pipeline"""
    pipeline_name: str
    model_name: str
    total_calls: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    success_rate: float = 1.0
    successful_extractions: int = 0
    failed_extractions: int = 0
    embedding_dim: int = None
    model_size_mb: float = 0.0
    load_time: float = 0.0
    
    def update(self, processing_time: float, success: bool):
        self.total_calls += 1
        self.total_processing_time += processing_time
        self.avg_processing_time = self.total_processing_time / self.total_calls
        
        if success:
            self.successful_extractions += 1
        else:
            self.failed_extractions += 1
            
        self.success_rate = self.successful_extractions / self.total_calls

@dataclass
class SystemStats:
    """System resource monitoring"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    gpu_available: bool = False
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    platform_info: str = ""
    python_version: str = ""
    torch_version: str = ""
    
    def update(self):
        # CPU and Memory
        self.cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        self.memory_percent = memory.percent
        self.memory_used_gb = memory.used / (1024**3)
        self.memory_total_gb = memory.total / (1024**3)
        
        # GPU
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            try:
                self.gpu_memory_used_mb = torch.cuda.memory_allocated() / (1024**2)
                self.gpu_memory_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            except:
                self.gpu_memory_used_mb = 0.0
                self.gpu_memory_total_mb = 0.0
        
        # System info
        self.platform_info = f"{platform.system()} {platform.release()}"
        self.python_version = platform.python_version()
        self.torch_version = torch.__version__

class DualPipelineAudioSearch:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # MODEL SELECTIONS - Using proven, reliable models
        self.text_embedder_name = "all-MiniLM-L6-v2"  # 384D, fast and efficient
        self.asr_model_name = "openai/whisper-base"     # Proven ASR model  
        self.audio_caption_model_name = "cahya/whisper-tiny-audio-captioning-v2.0"  # Working audio analysis
        
        # Models (loaded on demand)
        self.text_embedder = None
        self.asr_pipeline = None
        self.audio_caption_processor = None
        self.audio_caption_model = None
        
        # Statistics tracking
        self.stats = {
            'asr_pipeline': PipelineStats("ASR Pipeline", self.asr_model_name),
            'audio_pipeline': PipelineStats("Audio Analysis Pipeline", self.audio_caption_model_name),
            'text_embedder': PipelineStats("Text Embedder", self.text_embedder_name),
            'search_pipeline': PipelineStats("Search Pipeline", "Cosine Similarity")
        }
        
        # System monitoring
        self.system_stats = SystemStats()
        self.system_stats.update()
        
        # Audio database
        self.audio_segments = []
        
        # Model information for display
        self.model_info = {
            'text_embedder': {
                'name': 'all-MiniLM-L6-v2',
                'type': 'Sentence Transformer',
                'size': '90MB',
                'dimensions': '384D',
                'description': 'Fast and efficient sentence embeddings'
            },
            'asr_model': {
                'name': 'openai/whisper-base',
                'type': 'Speech Recognition',
                'size': '74MB',
                'dimensions': 'Audio → Text',
                'description': 'Proven ASR for speech transcription'
            },
            'audio_caption': {
                'name': 'cahya/whisper-tiny-audio-captioning-v2.0',
                'type': 'Audio Analysis',
                'size': '39MB',
                'dimensions': 'Audio → Description',
                'description': 'Audio content description for non-speech'
            }
        }
        
    def load_all_models(self) -> bool:
        """Load all models with detailed feedback"""
        st.info("🚀 Loading models - prioritizing reliability over speed...")
        
        success = True
        
        # 1. Text Embedder (most critical - everything depends on this)
        if self.text_embedder is None:
            with st.spinner("📝 Loading text embedder (all-MiniLM-L6-v2)..."):
                start_time = time.time()
                try:
                    self.text_embedder = SentenceTransformer(self.text_embedder_name)
                    load_time = time.time() - start_time
                    
                    # Get embedding dimension and calculate model size
                    test_embedding = self.text_embedder.encode("test")
                    self.stats['text_embedder'].embedding_dim = len(test_embedding)
                    self.stats['text_embedder'].load_time = load_time
                    
                    # Estimate model size
                    try:
                        model_size = sum(p.numel() * p.element_size() for p in self.text_embedder.parameters()) / (1024**2)
                        self.stats['text_embedder'].model_size_mb = model_size
                    except:
                        self.stats['text_embedder'].model_size_mb = 420  # Known size
                    
                    st.success(f"✅ Text embedder loaded ({load_time:.1f}s, {self.stats['text_embedder'].embedding_dim}D, {self.stats['text_embedder'].model_size_mb:.0f}MB)")
                except Exception as e:
                    st.error(f"❌ CRITICAL: Text embedder failed to load: {e}")
                    return False
        
        # 2. ASR Pipeline (Pipeline 1)
        if self.asr_pipeline is None:
            with st.spinner("🎤 Loading ASR pipeline (whisper-base)..."):
                start_time = time.time()
                try:
                    self.asr_pipeline = pipeline(
                        "automatic-speech-recognition",
                        model=self.asr_model_name,
                        device=0 if torch.cuda.is_available() else -1,
                        return_timestamps=False,
                        chunk_length_s=10,  # Handle longer audio chunks
                        stride_length_s=2   # Overlap for better accuracy
                    )
                    load_time = time.time() - start_time
                    self.stats['asr_pipeline'].load_time = load_time
                    self.stats['asr_pipeline'].model_size_mb = 74  # Known Whisper-base size
                    
                    st.success(f"✅ ASR pipeline loaded ({load_time:.1f}s, {self.stats['asr_pipeline'].model_size_mb}MB)")
                except Exception as e:
                    st.error(f"❌ ASR pipeline failed to load: {e}")
                    success = False
        
        # 3. Audio Caption Model (Pipeline 2)  
        if self.audio_caption_model is None:
            with st.spinner("🎵 Loading audio analysis model (cahya/whisper-tiny-audio-captioning-v2.0)..."):
                start_time = time.time()
                try:
                    self.audio_caption_processor = WhisperProcessor.from_pretrained(self.audio_caption_model_name)
                    self.audio_caption_model = WhisperForConditionalGeneration.from_pretrained(self.audio_caption_model_name)
                    self.audio_caption_model.to(self.device)
                    load_time = time.time() - start_time
                    self.stats['audio_pipeline'].load_time = load_time
                    
                    # Calculate model size
                    try:
                        model_size = sum(p.numel() * p.element_size() for p in self.audio_caption_model.parameters()) / (1024**2)
                        self.stats['audio_pipeline'].model_size_mb = model_size
                    except:
                        self.stats['audio_pipeline'].model_size_mb = 39  # Known size
                    
                    st.success(f"✅ Audio analysis model loaded ({load_time:.1f}s, {self.stats['audio_pipeline'].model_size_mb:.0f}MB)")
                except Exception as e:
                    st.error(f"❌ Audio analysis model failed to load: {e}")
                    success = False
        
        if success:
            st.success("🎉 All models loaded successfully!")
        
        return success
    
    def process_audio_file(self, audio_file) -> List[Dict]:
        """Process audio through both pipelines separately"""
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Load audio with better preprocessing for vocals
            audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
            
            # Less aggressive normalization - preserve dynamics
            # Only normalize if the audio is very quiet or very loud
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude < 0.1:  # Very quiet, boost it
                audio = audio * (0.5 / max_amplitude)
            elif max_amplitude > 0.95:  # Very loud, reduce it
                audio = audio * (0.8 / max_amplitude)
            # Otherwise leave it as-is
            
            st.info(f"📊 Audio loaded: {len(audio)/sr:.1f}s duration, {sr}Hz sample rate, max amplitude: {max_amplitude:.3f}")
            
            # Create longer segments (10-second chunks for better ASR context)
            segment_length = 10 * sr
            segments = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_segments = len(range(0, len(audio), segment_length))
            
            for i, start_idx in enumerate(range(0, len(audio), segment_length)):
                segment_audio = audio[start_idx:start_idx + segment_length]
                
                # Skip very short segments (less than 3 seconds)
                if len(segment_audio) < 3 * sr:
                    continue
                
                start_time = start_idx / sr
                end_time = min((start_idx + segment_length) / sr, len(audio) / sr)
                
                status_text.text(f"Processing segment {i+1}/{total_segments}: {start_time:.1f}s-{end_time:.1f}s")
                
                # PIPELINE 1: ASR Processing
                asr_text, asr_embedding = self._process_asr_pipeline(segment_audio)
                
                # PIPELINE 2: General Audio Analysis
                audio_description, audio_embedding = self._process_audio_pipeline(segment_audio, sr)
                
                # Only store if at least one pipeline succeeded
                if asr_text.strip() or audio_description.strip():
                    segments.append({
                        'segment_id': f"seg_{len(segments)}",
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time,
                        
                        # ASR Pipeline Results
                        'asr_text': asr_text,
                        'asr_embedding': asr_embedding,
                        'asr_success': len(asr_text.strip()) > 0,
                        
                        # Audio Analysis Pipeline Results  
                        'audio_description': audio_description,
                        'audio_embedding': audio_embedding,
                        'audio_success': len(audio_description.strip()) > 0,
                        
                        # Raw audio data for playback
                        'audio_data': segment_audio,
                        'sample_rate': sr
                    })
                
                # Update progress
                progress_bar.progress((i + 1) / total_segments)
            
            progress_bar.empty()
            status_text.empty()
            
            return segments
            
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def _process_asr_pipeline(self, audio: np.ndarray) -> Tuple[str, np.ndarray]:
        """Pipeline 1: ASR -> Text -> Embedding"""
        start_time = time.time()
        
        try:
            # ASR Processing with better parameters for music/vocals
            result = self.asr_pipeline(
                audio,
                generate_kwargs={
                    "language": "en",
                    "task": "transcribe",
                    "temperature": 0.2,  # Allow some randomness for better transcription
                    "no_repeat_ngram_size": 2,  # Less restrictive
                    "length_penalty": 0.8,  # Encourage shorter outputs
                    "repetition_penalty": 1.05,  # Gentle repetition penalty
                    "do_sample": True,  # Enable sampling
                    "num_beams": 1  # Faster processing
                }
            )
            
            # Extract text
            asr_text = result["text"].strip() if isinstance(result, dict) else str(result).strip()
            
            # Clean and validate ASR text with debugging
            if self._validate_asr_text(asr_text):
                # Create embedding
                embedding = self.text_embedder.encode(asr_text)
                processing_time = time.time() - start_time
                self.stats['asr_pipeline'].update(processing_time, success=True)
                print(f"✅ ASR Success: '{asr_text}' ({len(asr_text)} chars)")
                return asr_text, embedding
            else:
                processing_time = time.time() - start_time
                self.stats['asr_pipeline'].update(processing_time, success=False)
                print(f"❌ ASR Filtered: '{asr_text}' (failed validation)")
                return "", None
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats['asr_pipeline'].update(processing_time, success=False)
            print(f"ASR Pipeline Error: {e}")
            return "", None
    
    def _process_audio_pipeline(self, audio: np.ndarray, sr: int) -> Tuple[str, np.ndarray]:
        """Pipeline 2: Audio Analysis -> Description -> Embedding"""
        start_time = time.time()
        
        try:
            # Audio analysis processing
            input_features = self.audio_caption_processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate audio description
            with torch.no_grad():
                predicted_ids = self.audio_caption_model.generate(
                    input_features,
                    max_length=100,
                    no_repeat_ngram_size=3,
                    do_sample=False,
                    num_beams=2,  # Slightly better quality
                    repetition_penalty=1.3,
                    length_penalty=1.0,
                    early_stopping=True
                )
                
                audio_description = self.audio_caption_processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0].strip()
            
            # Clean and validate description
            if self._validate_audio_description(audio_description):
                # Create embedding
                embedding = self.text_embedder.encode(audio_description)
                processing_time = time.time() - start_time
                self.stats['audio_pipeline'].update(processing_time, success=True)
                return audio_description, embedding
            else:
                processing_time = time.time() - start_time  
                self.stats['audio_pipeline'].update(processing_time, success=False)
                return "", None
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats['audio_pipeline'].update(processing_time, success=False)
            print(f"Audio Pipeline Error: {e}")
            return "", None
    
    def _validate_asr_text(self, text: str) -> bool:
        """Validate ASR output quality - relaxed for vocals"""
        if not text or len(text.strip()) < 2:  # Allow shorter text
            return False
        
        text_stripped = text.strip()
        
        # Filter only obvious garbage patterns (more permissive)
        garbage_patterns = [
            'laionionion', 'ononon', 'lalala' * 3,  # Clear repetitive garbage
        ]
        
        text_lower = text_stripped.lower()
        
        # Check for obvious garbage
        for pattern in garbage_patterns:
            if pattern in text_lower:
                print(f"🗑️ ASR rejected for garbage pattern '{pattern}': {text}")
                return False
        
        # Allow more variation - only reject if it's mostly non-alphanumeric
        alphanumeric_chars = sum(1 for c in text_stripped if c.isalnum())
        total_chars = len(text_stripped)
        
        if total_chars > 0 and alphanumeric_chars / total_chars < 0.2:  # Very permissive
            print(f"🗑️ ASR rejected for low alphanumeric ratio ({alphanumeric_chars}/{total_chars}): {text}")
            return False
        
        # Additional checks for very short text
        if len(text_stripped) <= 3:
            # Very short text - be more selective
            if text_lower in ['um', 'uh', 'ah', 'eh', 'oh', 'mm']:
                print(f"🗑️ ASR rejected short filler word: {text}")
                return False
        
        print(f"✅ ASR accepted: '{text_stripped}' ({len(text_stripped)} chars, {alphanumeric_chars} alphanum)")
        return True
    
    def _validate_audio_description(self, description: str) -> bool:
        """Validate audio description quality"""
        if not description or len(description) < 5:
            return False
            
        # Filter garbage patterns
        if 'laionionion' in description.lower():
            return False
            
        # Should contain meaningful audio descriptors
        audio_keywords = [
            'music', 'sound', 'audio', 'playing', 'drums', 'guitar', 'piano',
            'singing', 'voice', 'speaking', 'noise', 'ambient', 'instrumental'
        ]
        
        description_lower = description.lower()
        has_audio_content = any(keyword in description_lower for keyword in audio_keywords)
        
        return has_audio_content
    
    def _analyze_query_for_weights(self, query: str) -> Tuple[float, float, str]:
        """Intelligently determine weights based on query keywords"""
        query_lower = query.lower()
        
        # ASR-focused keywords (speech, lyrics, words, names, communication)
        asr_keywords = [
            # Basic speech verbs
            'say', 'says', 'said', 'speak', 'speaking', 'spoke', 'spoken', 'talk', 'talking', 'talked',
            'tell', 'tells', 'told', 'mention', 'mentions', 'mentioned', 'call', 'calls', 'called',
            'shout', 'shouting', 'whisper', 'whispering', 'yell', 'yelling', 'scream', 'screaming',
            'cry', 'crying', 'laugh', 'laughing', 'giggle', 'giggling', 'chuckle', 'chuckling',
            
            # Words and language
            'word', 'words', 'phrase', 'phrases', 'sentence', 'sentences', 'language', 'languages',
            'text', 'message', 'messages', 'statement', 'statements', 'question', 'questions',
            'answer', 'answers', 'response', 'responses', 'reply', 'replies', 'comment', 'comments',
            
            # Vocals and singing
            'lyric', 'lyrics', 'sing', 'singing', 'sang', 'sung', 'song', 'songs', 'verse', 'verses',
            'chorus', 'choruses', 'bridge', 'vocal', 'vocals', 'vocalist', 'singer', 'singers',
            'voice', 'voices', 'voicing', 'harmony', 'harmonies', 'soprano', 'alto', 'tenor', 'bass',
            'choir', 'choral', 'anthem', 'ballad', 'opera', 'operatic', 'aria', 'lullaby',
            
            # Names and identity
            'name', 'names', 'named', 'title', 'titles', 'called', 'known', 'identify', 'identifies',
            'person', 'people', 'individual', 'character', 'characters', 'speaker', 'speakers',
            
            # Communication contexts
            'quote', 'quotes', 'quoted', 'announce', 'announces', 'announced', 'announcement',
            'dialogue', 'dialog', 'conversation', 'conversations', 'interview', 'interviews',
            'speech', 'speeches', 'presentation', 'presentations', 'lecture', 'lectures',
            'narrator', 'narration', 'narrating', 'reading', 'read', 'recite', 'reciting',
            'broadcast', 'broadcasting', 'podcast', 'podcasting', 'radio', 'commentary',
            'discussion', 'discussions', 'debate', 'debates', 'arguing', 'argument',
            
            # Speech characteristics
            'clear', 'clearly', 'articulate', 'articulated', 'mumble', 'mumbling', 'slur', 'slurring',
            'accent', 'accented', 'pronunciation', 'pronounce', 'enunciate', 'enunciation',
            'fluent', 'fluently', 'eloquent', 'eloquently', 'coherent', 'coherently',
            
            # Audio communication
            'microphone', 'mic', 'recording', 'recorded', 'voiceover', 'voicemail', 'telephone',
            'phone', 'call', 'calling', 'greeting', 'introduction', 'farewell', 'goodbye'
        ]
        
        # Audio-focused keywords (instruments, sounds, music characteristics, audio qualities)
        audio_keywords = [
            # Basic music terms
            'music', 'musical', 'musician', 'musicians', 'sound', 'sounds', 'sounding', 'audio',
            'sonic', 'acoustics', 'acoustic', 'acoustically', 'instrument', 'instrumental', 'instrumentation',
            
            # Rhythm and tempo
            'beat', 'beats', 'beating', 'rhythm', 'rhythmic', 'rhythmically', 'pulse', 'pulsing',
            'tempo', 'time', 'timing', 'meter', 'metrical', 'groove', 'groovy', 'swing', 'swinging',
            'syncopated', 'syncopation', 'polyrhythm', 'polyrhythmic', 'cross-rhythm',
            
            # Speed descriptors
            'fast', 'faster', 'fastest', 'quick', 'quicker', 'quickest', 'rapid', 'rapidly',
            'slow', 'slower', 'slowest', 'sluggish', 'crawling', 'moderate', 'medium',
            'accelerating', 'decelerating', 'speeding', 'slowing', 'rushing', 'dragging',
            
            # Volume and dynamics
            'loud', 'louder', 'loudest', 'quiet', 'quieter', 'quietest', 'soft', 'softer', 'softest',
            'silent', 'silence', 'mute', 'muted', 'whisper', 'whispering', 'booming', 'thunderous',
            'deafening', 'piercing', 'gentle', 'delicate', 'subtle', 'powerful', 'weak', 'strong',
            'crescendo', 'diminuendo', 'fortissimo', 'pianissimo', 'forte', 'piano', 'mezzo',
            
            # Tonal qualities
            'high', 'higher', 'highest', 'low', 'lower', 'lowest', 'deep', 'deeper', 'deepest',
            'sharp', 'flat', 'bright', 'dark', 'warm', 'cold', 'rich', 'thin', 'thick',
            'smooth', 'rough', 'harsh', 'sweet', 'bitter', 'metallic', 'wooden', 'glassy',
            'resonant', 'muffled', 'clear', 'muddy', 'crisp', 'fuzzy', 'clean', 'dirty',
            
            # Musical characteristics
            'melody', 'melodic', 'melodious', 'tune', 'tuning', 'tuned', 'harmony', 'harmonic', 'harmonious',
            'chord', 'chords', 'progression', 'scale', 'scales', 'key', 'major', 'minor',
            'note', 'notes', 'pitch', 'pitches', 'tone', 'tones', 'interval', 'intervals',
            'octave', 'fifth', 'fourth', 'third', 'second', 'seventh', 'ninth', 'eleventh', 'thirteenth',
            
            # Mood and energy
            'upbeat', 'downbeat', 'energetic', 'energy', 'lively', 'vibrant', 'dynamic', 'exciting',
            'calm', 'calming', 'peaceful', 'serene', 'tranquil', 'relaxing', 'soothing', 'meditative',
            'aggressive', 'intense', 'powerful', 'heavy', 'light', 'airy', 'ethereal', 'dreamy',
            'mysterious', 'ominous', 'cheerful', 'happy', 'sad', 'melancholy', 'nostalgic', 'romantic',
            'dramatic', 'epic', 'triumphant', 'victorious', 'heroic', 'majestic', 'grand',
            
            # Instruments - Strings
            'guitar', 'guitars', 'acoustic guitar', 'electric guitar', 'bass', 'bass guitar', 'upright bass',
            'violin', 'violins', 'viola', 'violas', 'cello', 'cellos', 'double bass', 'contrabass',
            'harp', 'harps', 'banjo', 'banjos', 'mandolin', 'mandolins', 'ukulele', 'ukuleles',
            'sitar', 'sitars', 'lute', 'lutes', 'strings', 'string section', 'bowed', 'plucked', 'strummed',
            
            # Instruments - Percussion
            'drum', 'drums', 'drumming', 'drummer', 'drumset', 'kit', 'snare', 'kick', 'hi-hat', 'hihat',
            'cymbal', 'cymbals', 'crash', 'ride', 'splash', 'tom', 'toms', 'timpani', 'timpanist',
            'percussion', 'percussive', 'percussionist', 'tambourine', 'triangle', 'cowbell',
            'conga', 'congas', 'bongo', 'bongos', 'djembe', 'tabla', 'cajon', 'marimba', 'xylophone',
            
            # Instruments - Wind/Brass
            'saxophone', 'sax', 'trumpet', 'trumpets', 'trombone', 'trombones', 'french horn', 'horn',
            'tuba', 'tubas', 'flute', 'flutes', 'clarinet', 'clarinets', 'oboe', 'oboes',
            'bassoon', 'bassoons', 'piccolo', 'recorder', 'harmonica', 'accordion', 'bagpipes',
            'brass', 'brass section', 'woodwind', 'woodwinds', 'wind', 'winds',
            
            # Instruments - Keys/Electronic
            'piano', 'pianos', 'keyboard', 'keyboards', 'organ', 'organs', 'harpsichord', 'synthesizer',
            'synth', 'synthesizers', 'electronic', 'digital', 'midi', 'sampler', 'sequencer',
            'drum machine', 'beats', 'loop', 'loops', 'sample', 'samples', 'vocoder', 'autotune',
            
            # Genres and styles
            'classical', 'baroque', 'romantic', 'contemporary', 'jazz', 'blues', 'rock', 'pop',
            'hip-hop', 'rap', 'country', 'folk', 'bluegrass', 'gospel', 'soul', 'funk', 'disco',
            'reggae', 'ska', 'punk', 'metal', 'grunge', 'alternative', 'indie', 'electronic',
            'techno', 'house', 'trance', 'ambient', 'drone', 'experimental', 'avant-garde',
            'world', 'ethnic', 'traditional', 'orchestral', 'symphonic', 'chamber', 'solo',
            
            # Audio effects and production
            'reverb', 'echo', 'delay', 'chorus', 'flanger', 'phaser', 'distortion', 'overdrive',
            'compression', 'limiter', 'equalizer', 'filter', 'filtering', 'boost', 'cut',
            'pan', 'panning', 'stereo', 'mono', 'surround', 'spatial', 'depth', 'width',
            'studio', 'recording', 'production', 'mix', 'mixing', 'mastered', 'mastering',
            
            # Environmental and ambient sounds
            'noise', 'background', 'foreground', 'ambient', 'atmosphere', 'atmospheric', 'environment',
            'natural', 'artificial', 'synthetic', 'processed', 'raw', 'live', 'recorded',
            'field recording', 'soundscape', 'texture', 'layer', 'layers', 'overdub', 'multitrack'
        ]
        
        # Count keyword matches
        asr_matches = sum(1 for keyword in asr_keywords if keyword in query_lower)
        audio_matches = sum(1 for keyword in audio_keywords if keyword in query_lower)
        
        # Smart weight calculation with constrained range
        # Base weights: 40%-60% range (never goes to extremes)
        # Can shift to 20%-80% for strong keyword bias
        
        if asr_matches == 0 and audio_matches == 0:
            # No specific keywords, use balanced weights
            asr_weight, audio_weight = 0.5, 0.5
            analysis = "Balanced (no specific keywords detected)"
        
        elif asr_matches > 0 and audio_matches == 0:
            # Pure ASR query
            strength = min(asr_matches / 3.0, 1.0)  # Cap influence
            asr_weight = 0.5 + (0.3 * strength)  # Range: 0.5 to 0.8
            audio_weight = 1.0 - asr_weight
            analysis = f"ASR-focused ({asr_matches} speech keywords)"
        
        elif audio_matches > 0 and asr_matches == 0:
            # Pure audio query
            strength = min(audio_matches / 3.0, 1.0)  # Cap influence
            audio_weight = 0.5 + (0.3 * strength)  # Range: 0.5 to 0.8
            asr_weight = 1.0 - audio_weight
            analysis = f"Audio-focused ({audio_matches} audio keywords)"
        
        else:
            # Mixed query - proportional weighting with constraints
            total_keywords = asr_matches + audio_matches
            asr_ratio = asr_matches / total_keywords
            
            # Map ratio to constrained range (0.2 to 0.8)
            asr_weight = 0.2 + (asr_ratio * 0.6)
            audio_weight = 1.0 - asr_weight
            analysis = f"Mixed query (ASR:{asr_matches}, Audio:{audio_matches})"
        
        return asr_weight, audio_weight, analysis
    
    def search_with_fusion(self, query: str) -> Tuple[List[Dict], Dict]:
        """Search using intelligent query-based weighted fusion"""
        if not self.audio_segments:
            return [], {}
        
        start_time = time.time()
        
        # Analyze query to determine optimal weights
        asr_weight, audio_weight, weight_analysis = self._analyze_query_for_weights(query)
        
        # Create query embedding
        query_embedding = self.text_embedder.encode(query).reshape(1, -1)
        
        results = []
        
        for segment in self.audio_segments:
            asr_similarity = 0.0
            audio_similarity = 0.0
            
            # Calculate ASR similarity
            if segment['asr_embedding'] is not None:
                asr_embedding = segment['asr_embedding'].reshape(1, -1)
                asr_similarity = float(cosine_similarity(query_embedding, asr_embedding)[0][0])
            
            # Calculate audio description similarity  
            if segment['audio_embedding'] is not None:
                audio_embedding = segment['audio_embedding'].reshape(1, -1)
                audio_similarity = float(cosine_similarity(query_embedding, audio_embedding)[0][0])
            
            # Apply intelligent weighted fusion
            if asr_similarity > 0 or audio_similarity > 0:
                # Use data availability to adjust weights
                effective_asr_weight = asr_weight if segment['asr_success'] else 0
                effective_audio_weight = audio_weight if segment['audio_success'] else 0
                
                total_weight = effective_asr_weight + effective_audio_weight
                
                if total_weight > 0:
                    # Normalize weights
                    effective_asr_weight /= total_weight
                    effective_audio_weight /= total_weight
                    
                    # Calculate weighted average
                    fusion_score = (
                        effective_asr_weight * asr_similarity +
                        effective_audio_weight * audio_similarity
                    )
                    
                    if fusion_score > 0.1:  # Relevance threshold
                        results.append({
                            **segment,
                            'asr_similarity': asr_similarity,
                            'audio_similarity': audio_similarity,
                            'fusion_score': fusion_score,
                            'effective_asr_weight': effective_asr_weight,
                            'effective_audio_weight': effective_audio_weight,
                            'query_asr_weight': asr_weight,
                            'query_audio_weight': audio_weight
                        })
        
        # Sort by fusion score
        results.sort(key=lambda x: x['fusion_score'], reverse=True)
        
        # Update search stats
        processing_time = time.time() - start_time
        self.stats['search_pipeline'].update(processing_time, success=len(results) > 0)
        
        # Return results and weight analysis
        weight_info = {
            'asr_weight': asr_weight,
            'audio_weight': audio_weight,
            'analysis': weight_analysis,
            'query': query
        }
        
        return results[:10], weight_info  # Top 10 results

# Streamlit Interface
def main():
    st.set_page_config(page_title="Dual Pipeline Audio Search", layout="wide")
    st.title("🎯 Dual Pipeline Audio Search System")
    st.caption("ASR + General Audio Analysis with Weighted Fusion")
    
    # Initialize system
    if 'search_system' not in st.session_state:
        st.session_state.search_system = DualPipelineAudioSearch()
    
    search_system = st.session_state.search_system
    
    # Sidebar - System Status & Configuration
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Update system stats
        search_system.system_stats.update()
        
        # Model loading
        if st.button("🚀 Load All Models", type="primary"):
            success = search_system.load_all_models()
            if success:
                st.balloons()
        
        st.divider()
        
        # System Resources (Real-time)
        st.subheader("💻 System Resources")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU", f"{search_system.system_stats.cpu_percent:.1f}%")
            st.metric("Memory", f"{search_system.system_stats.memory_used_gb:.1f}GB", 
                     f"{search_system.system_stats.memory_percent:.1f}%")
        with col2:
            if search_system.system_stats.gpu_available:
                st.metric("GPU Memory", f"{search_system.system_stats.gpu_memory_used_mb:.0f}MB",
                         f"{search_system.system_stats.gpu_memory_used_mb/search_system.system_stats.gpu_memory_total_mb*100:.1f}%")
            else:
                st.metric("GPU", "CPU Mode")
            
            st.metric("Database", f"{len(search_system.audio_segments)} segments")
        
        st.divider()
        
        # Intelligent Fusion Weights Info
        st.subheader("🧠 Intelligent Fusion")
        st.info("Weights automatically adjust based on your query keywords!")
        
        with st.expander("📖 How It Works"):
            st.write("**ASR Keywords:** speak, say, lyrics, voice, words, etc.")
            st.write("**Audio Keywords:** music, drums, guitar, melody, sound, etc.")
            st.write("**Weight Range:** 20% - 80% (never extreme)")
        
        st.divider()
        
        # Quick Stats
        st.subheader("⚡ Quick Stats")
        total_calls = sum(stats.total_calls for stats in search_system.stats.values())
        total_time = sum(stats.total_processing_time for stats in search_system.stats.values())
        
        if total_calls > 0:
            st.metric("Total Operations", total_calls)
            st.metric("Total Processing", f"{total_time:.1f}s")
            st.metric("Avg Per Operation", f"{total_time/total_calls:.3f}s")
    
    # Main interface with statistics tab
    tab1, tab2, tab3 = st.tabs(["📁 Process Audio", "🔍 Search", "📊 Statistics"])
    
    with tab1:
        st.header("Audio Processing")
        st.write("Upload audio files to process through both ASR and audio analysis pipelines")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Audio will be processed through both ASR and general audio analysis"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.audio(uploaded_file)
            
            with col2:
                st.info(f"**Models Status:**\n"
                       f"- Text Embedder: {'✅' if search_system.text_embedder else '❌'}\n"
                       f"- ASR Pipeline: {'✅' if search_system.asr_pipeline else '❌'}\n"
                       f"- Audio Analysis: {'✅' if search_system.audio_caption_model else '❌'}")
            
            if st.button("🔄 Process with Both Pipelines", type="primary"):
                if all([search_system.text_embedder, search_system.asr_pipeline, search_system.audio_caption_model]):
                    with st.spinner("Processing through both pipelines..."):
                        try:
                            segments = search_system.process_audio_file(uploaded_file)
                            search_system.audio_segments.extend(segments)
                            
                            # Show results summary
                            st.success(f"✅ Processed {len(segments)} segments")
                            
                            # Detailed breakdown
                            asr_successes = sum(1 for s in segments if s['asr_success'])
                            audio_successes = sum(1 for s in segments if s['audio_success'])
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Segments", len(segments))
                            with col2:
                                st.metric("ASR Successes", asr_successes, f"{asr_successes/len(segments):.1%}")
                            with col3:
                                st.metric("Audio Analysis Successes", audio_successes, f"{audio_successes/len(segments):.1%}")
                            
                        except Exception as e:
                            st.error(f"❌ Processing failed: {e}")
                else:
                    st.error("⚠️ Please load all models first using the sidebar button")
    
    with tab2:
        st.header("Weighted Fusion Search")
        
        if search_system.audio_segments:
            st.write(f"Database: {len(search_system.audio_segments)} segments ready for search")
            
            query = st.text_input(
                "Search Query",
                placeholder="e.g., 'upbeat music with drums', 'person speaking clearly', 'guitar solo'"
            )
            
            if query and st.button("🎯 Search with Fusion", type="primary"):
                with st.spinner("Searching with intelligent weighted fusion..."):
                    results, weight_info = search_system.search_with_fusion(query)
                    
                    # Show intelligent weight analysis
                    st.subheader("🧠 Intelligent Weight Analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("ASR Weight", f"{weight_info['asr_weight']:.1%}")
                    with col2:
                        st.metric("Audio Weight", f"{weight_info['audio_weight']:.1%}")
                    with col3:
                        st.info(f"**Analysis:** {weight_info['analysis']}")
                    
                    if results:
                        st.success(f"🎯 Found {len(results)} relevant segments")
                        
                        for i, result in enumerate(results):
                            with st.expander(f"Result {i+1}: Fusion Score {result['fusion_score']:.3f}"):
                                
                                # Score breakdown
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("ASR Score", f"{result['asr_similarity']:.3f}")
                                with col2:
                                    st.metric("Audio Score", f"{result['audio_similarity']:.3f}")
                                with col3:
                                    st.metric("Fusion Score", f"{result['fusion_score']:.3f}")
                                
                                # Content
                                st.write(f"**Time:** {result['start_time']:.1f}s - {result['end_time']:.1f}s")
                                
                                if result['asr_text']:
                                    st.write(f"**🎤 ASR Text:** {result['asr_text']}")
                                
                                if result['audio_description']:
                                    st.write(f"**🎵 Audio Description:** {result['audio_description']}")
                                
                                # Weights used (both query-level and effective)
                                st.caption(f"Query Weights: ASR={result['query_asr_weight']:.1%}, Audio={result['query_audio_weight']:.1%}")
                                st.caption(f"Effective Weights: ASR={result['effective_asr_weight']:.1%}, Audio={result['effective_audio_weight']:.1%}")
                                
                                # Audio playback
                                st.audio(result['audio_data'], sample_rate=result['sample_rate'])
                    
                    else:
                        st.info("No relevant segments found. Try different keywords - the system will automatically adjust weights!")
        
        else:
            st.info("👆 Process some audio files first to enable search")
    
    with tab3:
        st.header("📊 Comprehensive Statistics")
        
        # Model Information
        st.subheader("🤖 Model Information")
        model_cols = st.columns(3)
        
        with model_cols[0]:
            st.markdown("### 📝 Text Embedder")
            info = search_system.model_info['text_embedder']
            st.write(f"**Model:** {info['name']}")
            st.write(f"**Type:** {info['type']}")
            st.write(f"**Size:** {info['size']}")
            st.write(f"**Output:** {info['dimensions']}")
            st.caption(info['description'])
            
            if search_system.stats['text_embedder'].total_calls > 0:
                stats = search_system.stats['text_embedder']
                st.metric("Load Time", f"{stats.load_time:.2f}s")
                st.metric("Calls", stats.total_calls)
                st.metric("Avg Time", f"{stats.avg_processing_time:.3f}s")
        
        with model_cols[1]:
            st.markdown("### 🎤 ASR Model")
            info = search_system.model_info['asr_model']
            st.write(f"**Model:** {info['name']}")
            st.write(f"**Type:** {info['type']}")
            st.write(f"**Size:** {info['size']}")
            st.write(f"**Output:** {info['dimensions']}")
            st.caption(info['description'])
            
            if search_system.stats['asr_pipeline'].total_calls > 0:
                stats = search_system.stats['asr_pipeline']
                st.metric("Load Time", f"{stats.load_time:.2f}s")
                st.metric("Success Rate", f"{stats.success_rate:.1%}")
                st.metric("Avg Time", f"{stats.avg_processing_time:.3f}s")
        
        with model_cols[2]:
            st.markdown("### 🎵 Audio Analysis")
            info = search_system.model_info['audio_caption']
            st.write(f"**Model:** {info['name']}")
            st.write(f"**Type:** {info['type']}")
            st.write(f"**Size:** {info['size']}")
            st.write(f"**Output:** {info['dimensions']}")
            st.caption(info['description'])
            
            if search_system.stats['audio_pipeline'].total_calls > 0:
                stats = search_system.stats['audio_pipeline']
                st.metric("Load Time", f"{stats.load_time:.2f}s")
                st.metric("Success Rate", f"{stats.success_rate:.1%}")
                st.metric("Avg Time", f"{stats.avg_processing_time:.3f}s")
        
        st.divider()
        
        # System Information
        st.subheader("💻 System Information")
        sys_cols = st.columns(2)
        
        with sys_cols[0]:
            st.markdown("### Hardware")
            st.write(f"**Platform:** {search_system.system_stats.platform_info}")
            st.write(f"**CPU Usage:** {search_system.system_stats.cpu_percent:.1f}%")
            st.write(f"**Memory:** {search_system.system_stats.memory_used_gb:.1f}GB / {search_system.system_stats.memory_total_gb:.1f}GB ({search_system.system_stats.memory_percent:.1f}%)")
            
            if search_system.system_stats.gpu_available:
                st.write(f"**GPU:** Available ({search_system.system_stats.gpu_memory_used_mb:.0f}MB / {search_system.system_stats.gpu_memory_total_mb:.0f}MB)")
            else:
                st.write("**GPU:** Not available (CPU mode)")
        
        with sys_cols[1]:
            st.markdown("### Software")
            st.write(f"**Python:** {search_system.system_stats.python_version}")
            st.write(f"**PyTorch:** {search_system.system_stats.torch_version}")
            st.write(f"**Device:** {search_system.device}")
            st.write(f"**Audio Segments:** {len(search_system.audio_segments)}")
        
        st.divider()
        
        # Detailed Pipeline Statistics
        st.subheader("⚡ Detailed Pipeline Performance")
        
        if any(stats.total_calls > 0 for stats in search_system.stats.values()):
            perf_cols = st.columns(4)
            
            for i, (name, stats) in enumerate(search_system.stats.items()):
                if stats.total_calls > 0:
                    with perf_cols[i % 4]:
                        st.markdown(f"### {stats.pipeline_name}")
                        st.metric("Total Calls", stats.total_calls)
                        st.metric("Total Time", f"{stats.total_processing_time:.2f}s")
                        st.metric("Avg Time", f"{stats.avg_processing_time:.3f}s")
                        st.metric("Success Rate", f"{stats.success_rate:.1%}")
                        
                        if stats.successful_extractions > 0:
                            st.metric("Successful", stats.successful_extractions)
                        if stats.failed_extractions > 0:
                            st.metric("Failed", stats.failed_extractions)
                        
                        if stats.embedding_dim:
                            st.caption(f"Embedding: {stats.embedding_dim}D")
                        if stats.model_size_mb > 0:
                            st.caption(f"Model: {stats.model_size_mb:.0f}MB")
        else:
            st.info("📈 Process some audio files to see performance statistics!")
        
        st.divider()
        
        # Memory Management
        st.subheader("🧠 Memory Management")
        mem_cols = st.columns(3)
        
        with mem_cols[0]:
            if st.button("🗑️ Force Garbage Collection"):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                search_system.system_stats.update()
                st.success("Memory cleaned!")
        
        with mem_cols[1]:
            if st.button("🔄 Refresh System Stats"):
                search_system.system_stats.update()
                st.rerun()
        
        with mem_cols[2]:
            if st.button("📊 Export Statistics"):
                stats_data = {
                    'system': {
                        'platform': search_system.system_stats.platform_info,
                        'cpu_percent': search_system.system_stats.cpu_percent,
                        'memory_percent': search_system.system_stats.memory_percent,
                        'gpu_available': search_system.system_stats.gpu_available
                    },
                    'models': {name: {
                        'total_calls': stats.total_calls,
                        'avg_processing_time': stats.avg_processing_time,
                        'success_rate': stats.success_rate,
                        'model_size_mb': stats.model_size_mb
                    } for name, stats in search_system.stats.items()},
                    'database': {'total_segments': len(search_system.audio_segments)}
                }
                st.download_button(
                    "💾 Download JSON",
                    data=str(stats_data).replace("'", '"'),
                    file_name=f"audio_search_stats_{int(time.time())}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()