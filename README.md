# 🎵 ClipABit: Text-to-Audio Search Demo

A Streamlit application that demonstrates cross-modal audio search using AI embeddings. Upload audio files and search for them using natural language descriptions - no transcription needed!

## 🎯 What This Does

- **Cross-Modal Search**: Type "piano music" and find actual piano audio clips
- **No Speech Required**: Works with music, sound effects, nature sounds, etc.
- **AI-Powered**: Uses CLAP (Contrastive Language-Audio Pre-training) models
- **Real-Time**: Search through your audio database instantly
- **Model Comparison**: Easy switching between different embedding models
- **Performance Analytics**: Detailed metrics for every operation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (tested with Python 3.13)
- Windows/macOS/Linux

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Ethan-McManus-Projects/ClipABit-Pipeline-Testing-and-Iteration.git
cd ClipABit-Pipeline-Testing-and-Iteration
```

2. **Create a virtual environment**
```bash
# Windows
python -m venv audio_search_venv
audio_search_venv\Scripts\activate

# macOS/Linux  
python -m venv audio_search_venv
source audio_search_venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run streamlit_app.py
```

5. **Open in browser**
- The app will automatically open at `http://localhost:8501`
- Or manually navigate to the URL shown in terminal

## 🎮 How to Use

### Basic Workflow
1. **Upload Audio Files** 📁
   - Drag and drop MP3, WAV, M4A, or FLAC files
   - App automatically splits them into searchable chunks
   - Files are processed and embeddings are generated

2. **Search with Text** 🔍
   - Type natural language descriptions like:
     - `"upbeat music with drums"`
     - `"speech or talking"`
     - `"piano melody"`
     - `"bird sounds"`
     - `"guitar solo"`

3. **Listen to Results** 🎧
   - App returns ranked audio clips
   - Click play to hear each result
   - Similarity scores help evaluate matches

### Advanced Features

#### Model Switching 🔄
- Use sidebar dropdown to try different AI models
- Compare performance and accuracy
- Models available:
  - **CLAP-HTSAT (Recommended)**: Best balance of speed/accuracy
  - **CLAP-HTSAT-Fused**: Higher accuracy, slower processing
  - **Microsoft CLAP**: Optimized for speech and general audio

#### Performance Metrics 📊
- Enable "Show Performance Metrics" in sidebar for live stats
- Check "Advanced Metrics Dashboard" for detailed analysis
- Export metrics to CSV for further analysis
- Track:
  - Model loading times
  - Embedding generation speed
  - Search performance
  - Memory usage
  - Database statistics

#### Model Comparison Mode 🆚
- Use "Model Comparison Mode" for systematic testing
- Reset database to test same files with different models
- Compare accuracy and speed across models

## 🔧 Technical Details

### Architecture
- **Frontend**: Streamlit for interactive web interface
- **Audio Processing**: librosa for audio loading and chunking
- **AI Models**: HuggingFace Transformers (CLAP models)
- **Search**: Cosine similarity on embedding vectors
- **Storage**: In-memory vector database (suitable for demos)

### Supported Audio Formats
- MP3, WAV, M4A, FLAC
- Automatic resampling to 48kHz
- Configurable chunk duration (5-30 seconds)

### Embedding Models
All models create aligned text-audio embeddings in the same vector space, enabling cross-modal search:

- **laion/clap-htsat-unfused**: General purpose, 512-dimensional embeddings
- **laion/clap-htsat-fused**: Fused architecture, potentially better accuracy
- **microsoft/clap-htsat-unfused**: Microsoft's implementation

### Performance Characteristics
- **Model Loading**: ~10-30 seconds (first time only)
- **Audio Processing**: ~1-5 seconds per minute of audio
- **Embedding Generation**: ~100-500ms per 10-second chunk
- **Search**: <100ms for databases up to 1000 clips
- **Memory Usage**: ~500MB-2GB depending on model and database size

## 🛠️ Development

### Project Structure
```
ClipABit-Pipeline-Testing-and-Iteration/
├── streamlit_app.py          # Main application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── .gitignore               # Git ignore rules
└── audio_search_venv/       # Virtual environment (not in git)
```

### Key Components

#### Model Configuration
Models are easily configurable in `streamlit_app.py`:
```python
AVAILABLE_MODELS = {
    "Model Name": {
        "model_name": "huggingface/model-id",
        "description": "Model description"
    }
}
```

#### Metrics System
Comprehensive logging of all operations:
- Model loading and memory usage
- Audio processing pipelines  
- Embedding generation performance
- Search and retrieval timing
- System resource utilization

### Adding New Models
To add a new embedding model:

1. Add to `AVAILABLE_MODELS` dict in `streamlit_app.py`
2. Ensure model is compatible with CLAP processor interface
3. Test with model comparison mode

## 🔮 Future Plans

### Short Term (Next 2-4 weeks)
- [ ] **Persistent Storage**: Replace in-memory database with vector DB (FAISS/Chroma)
- [ ] **Batch Upload**: Process multiple files simultaneously  
- [ ] **Audio Visualization**: Waveform display for search results
- [ ] **Export Results**: Save search results and playlists
- [ ] **Docker Container**: Easy deployment and sharing

### Medium Term (1-2 months)  
- [ ] **Similarity Threshold**: Filter low-quality matches
- [ ] **Advanced Chunking**: Intelligent boundary detection
- [ ] **Multi-Modal**: Add image/video embedding support
- [ ] **API Interface**: REST API for programmatic access
- [ ] **Cloud Deployment**: Deploy to cloud platforms

### Long Term (2-6 months)
- [ ] **Custom Model Training**: Fine-tune embeddings for specific domains
- [ ] **Real-time Audio**: Live audio stream processing
- [ ] **Collaborative Features**: Share databases between users  
- [ ] **Advanced Analytics**: Search pattern analysis and recommendations
- [ ] **Mobile App**: iOS/Android companion app

## 🐛 Known Limitations

### Current Version (v0.1)
- **Memory Storage**: Database cleared on app restart
- **Single Session**: No multi-user support
- **Large Files**: May be slow with very long audio files (>10 minutes)
- **Model Size**: First-time model download requires good internet connection
- **Concurrent Users**: Not optimized for multiple simultaneous users

### Performance Notes
- **Initial Load**: First model load takes 10-30 seconds
- **Memory Usage**: Grows with database size (~1KB per audio chunk)
- **CPU Usage**: Embedding generation is CPU-intensive
- **GPU Support**: Currently CPU-only (GPU support planned)

## 📞 Getting Help

### Common Issues

**"StreamLit command not found"**
- Ensure virtual environment is activated
- Try: `python -m streamlit run streamlit_app.py`

**"Model download failed"**  
- Check internet connection
- Some models are 1-2GB and require stable connection
- Try switching to a smaller model first

**"Out of memory errors"**
- Reduce chunk duration in sidebar
- Process fewer/shorter audio files
- Close other applications

**"Audio processing errors"**
- Ensure audio files are valid format
- Try converting to WAV format first
- Check file permissions

### Contributing
This is a research/demo project. Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests for improvements
- Share interesting use cases or results

### Contact
- **Project**: ClipABit Pipeline Testing and Iteration
- **Repository**: https://github.com/Ethan-McManus-Projects/ClipABit-Pipeline-Testing-and-Iteration
- **Issues**: Use GitHub Issues for bug reports and feature requests

## 📝 License

This project is for research and educational purposes. See individual model licenses for usage restrictions.

## 🙏 Acknowledgments

- **LAION**: For the CLAP models and training
- **HuggingFace**: For the transformers library and model hosting
- **Streamlit**: For the incredible web app framework
- **librosa**: For audio processing capabilities

---

**Note for Collaborators**: This is an active research project. The codebase changes frequently as we experiment with different approaches. Check the GitHub repository for the latest updates and documentation.