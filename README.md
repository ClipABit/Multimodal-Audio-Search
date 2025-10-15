# 🎵 ClipABit Audio Search System

A production-ready dual pipeline audio search system that combines **Automatic Speech Recognition (ASR)** and **Audio Analysis** for comprehensive audio content discovery.

## ✨ Features

### 🔍 **Intelligent Dual Pipeline Search**
- **ASR Pipeline**: Uses OpenAI Whisper for speech transcription and vocal content
- **Audio Analysis Pipeline**: Uses specialized models for non-speech audio description
- **Intelligent Query Weighting**: Automatically adjusts pipeline weights based on query keywords

### 🎯 **Smart Query Processing**
- **Keyword-Based Weight Adjustment**: System analyzes your query and optimizes pipeline weights
- **Semantic Search**: Uses sentence transformers for meaningful content matching
- **Unified Embedding Space**: All content processed into consistent 384D embeddings

### 📊 **Comprehensive Analytics**
- **Real-time System Monitoring**: CPU, memory, and GPU usage tracking
- **Performance Statistics**: Model loading times, processing speeds, success rates
- **Model Information Dashboard**: Detailed specs for all AI models used

### ⚡ **Optimized Performance**
- **Fast all-MiniLM-L6-v2 Embeddings**: 90MB model, 4x faster than alternatives
- **10-Second Audio Segments**: Optimized for ASR accuracy
- **Efficient Processing**: Smart caching and memory management

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **Windows/macOS/Linux**
- **8GB+ RAM recommended**
- **Optional: CUDA-compatible GPU for acceleration**

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ethan-McManus-Projects/ClipABit-Pipeline-Testing-and-Iteration.git
   cd ClipABit-Pipeline-Testing-and-Iteration
   ```

2. **Create virtual environment**
   ```bash
   python -m venv audio_search_venv
   
   # Windows
   .\audio_search_venv\Scripts\activate
   
   # macOS/Linux
   source audio_search_venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   # Windows
   .\audio_search_venv\Scripts\python.exe -m streamlit run audio_search.py --server.port 8527
   
   # macOS/Linux
   python -m streamlit run audio_search.py --server.port 8527
   ```

5. **Open in browser**
   - Navigate to `http://localhost:8527`

## 📖 Usage Guide

### 1. **Upload Audio Files**
- Support formats: MP3, WAV, M4A, FLAC, OGG
- Maximum file size: 200MB per file
- Automatic 10-second segmentation for optimal processing

### 2. **Search Your Audio**
- Enter natural language queries like:
  - `"guitar solo"` - Finds instrumental guitar sections
  - `"sad vocals"` - Locates emotional vocal content
  - `"upbeat rhythm"` - Discovers energetic musical sections
  - `"conversation about technology"` - Finds specific speech topics

### 3. **Monitor Performance**
- **Statistics Tab**: View model information and system resources
- **Real-time Metrics**: CPU, memory, GPU usage
- **Performance Analytics**: Processing times and success rates

## 🧠 Technical Architecture

### **AI Models Used**
| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| **Text Embeddings** | all-MiniLM-L6-v2 | 90MB | Semantic search and query processing |
| **Speech Recognition** | OpenAI Whisper-base | 74MB | Transcribing vocal content |
| **Audio Analysis** | Whisper-tiny-audio-captioning | 39MB | Describing non-speech audio |

### **Intelligent Weighting System**
- **ASR Keywords**: Vocal-focused terms (lyrics, singing, speech, etc.)
- **Audio Keywords**: Instrumental terms (guitar, drums, melody, etc.)
- **Dynamic Range**: 20%-80% weight distribution based on query analysis
- **Default Fallback**: 50-50 split for ambiguous queries

### **Processing Pipeline**
1. **Audio Segmentation**: 10-second chunks with overlap
2. **Dual Processing**: Parallel ASR and audio analysis
3. **Embedding Generation**: Unified 384D semantic vectors
4. **Query Matching**: Cosine similarity search
5. **Intelligent Fusion**: Weighted result combination

## 🔧 Configuration

### **System Requirements**
- **Minimum RAM**: 4GB (8GB+ recommended)
- **Storage**: 2GB for models and dependencies
- **CPU**: Multi-core recommended for faster processing
- **GPU**: Optional CUDA support for acceleration

### **Performance Tuning**
- **Batch Processing**: Upload multiple files for efficiency
- **Memory Management**: Built-in garbage collection tools
- **Model Caching**: Models persist between sessions

## 📊 Statistics & Monitoring

The system provides comprehensive analytics:
- **Model Performance**: Load times, processing speeds, success rates
- **System Resources**: Real-time CPU, memory, GPU monitoring
- **Search Analytics**: Query processing times and result quality
- **Export Functionality**: Download performance data as JSON

## 🗂️ Project Structure

```
ClipABit-Pipeline-Testing-and-Iteration/
├── audio_search.py              # Main application (production ready)
├── requirements.txt             # Python dependencies
├── setup_windows.bat           # Windows setup script
├── setup_unix.sh              # Unix/Linux setup script
├── audio_search_venv/          # Virtual environment
├── previous_iterations/        # Historical development files
└── README.md                   # This file
```

## 🚧 Development History

All previous iterations and experimental versions are preserved in the `previous_iterations/` folder, including:
- Early single-pipeline prototypes
- Lightweight architecture experiments
- Alternative model implementations
- Deployment documentation

## 🤝 Contributing

This is a testing and iteration repository for the ClipABit audio search system. Feel free to:
- Report issues and bugs
- Suggest feature improvements
- Submit performance optimizations
- Share usage feedback

## 📝 License

This project is part of the ClipABit ecosystem. Please refer to the main project repository for licensing information.

## 🔗 Related Projects

- **ClipABit Main**: Core audio processing platform
- **ClipABit Extensions**: Additional audio analysis tools
- **ClipABit API**: RESTful service implementation

---

**Built with ❤️ for intelligent audio search and discovery**