# 🚀 Deployment Guide for Collaborators

This guide helps others run the ClipABit audio search demo on their machines.

## For End Users (Just Want to Try It)

### Option 1: Local Installation (Recommended)

**Prerequisites:**
- Python 3.8 or newer
- 4GB+ RAM (for AI models)  
- 2GB+ free disk space
- Stable internet (for initial model download)

**Steps:**
```bash
# 1. Clone the repository
git clone https://github.com/Ethan-McManus-Projects/ClipABit-Pipeline-Testing-and-Iteration.git
cd ClipABit-Pipeline-Testing-and-Iteration

# 2. Create virtual environment
python -m venv audio_search_venv

# 3. Activate virtual environment
# Windows:
audio_search_venv\Scripts\activate
# macOS/Linux:
source audio_search_venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run streamlit_app.py
```

**First Run Notes:**
- Model download takes 5-10 minutes (1-2GB files)
- App opens automatically at `http://localhost:8501`
- Upload some test audio files to start

### Option 2: Docker (Coming Soon)
We're working on a Docker container for even easier deployment.

## For Developers

### Development Setup
```bash
# Clone and setup as above, then:

# Install development dependencies
pip install -r requirements.txt
pip install jupyter notebook  # Optional for experimentation

# Run in development mode
streamlit run streamlit_app.py --server.runOnSave true
```

### Environment Variables
Create a `.env` file for configuration:
```env
# Model settings
DEFAULT_MODEL=laion/clap-htsat-unfused
CHUNK_DURATION=10
MAX_RESULTS=5

# Performance settings  
ENABLE_GPU=false
CACHE_DIR=./model_cache
```

### Adding New Models
1. Edit `AVAILABLE_MODELS` in `streamlit_app.py`
2. Test with model comparison mode
3. Submit PR with performance benchmarks

## Common Issues & Solutions

### "Streamlit not found"
```bash
# Ensure virtual environment is activated
# Windows:
audio_search_venv\Scripts\activate
# Then try:
python -m streamlit run streamlit_app.py
```

### "Model download failed"
- Check internet connection (models are 1-2GB)
- Try different model from dropdown
- Clear cache: `rm -rf ~/.cache/huggingface/`

### "Out of memory"
- Close other applications
- Reduce chunk duration in sidebar
- Try smaller model (Microsoft CLAP)

### "Audio processing errors"
- Check audio file format (MP3, WAV, M4A, FLAC)
- Try shorter files first
- Ensure file isn't corrupted

## Platform-Specific Notes

### Windows
- Use PowerShell or Command Prompt
- Virtual environment: `audio_search_venv\Scripts\activate`
- May need Visual Studio Build Tools for some dependencies

### macOS
- Use Terminal
- Virtual environment: `source audio_search_venv/bin/activate`
- May need Xcode Command Line Tools: `xcode-select --install`

### Linux
- Use any terminal
- Virtual environment: `source audio_search_venv/bin/activate`  
- May need build essentials: `sudo apt-get install build-essential`

## Resource Requirements

### Minimum Requirements
- **CPU**: 2+ cores, 2.0GHz+
- **RAM**: 4GB (8GB recommended)
- **Storage**: 5GB free space
- **Network**: Stable internet for model download

### Recommended Setup
- **CPU**: 4+ cores, 3.0GHz+
- **RAM**: 8GB+ (16GB for large databases)
- **Storage**: 10GB+ SSD
- **GPU**: Not required but will speed up future versions

## Performance Tips

### For Better Speed
1. Use SSD storage for model cache
2. Close unnecessary applications
3. Use recommended models (CLAP-HTSAT)
4. Process shorter audio chunks

### For Better Accuracy
1. Use fused models (slower but more accurate)
2. Upload diverse audio types for testing
3. Use descriptive search terms
4. Enable metrics to track performance

## Sharing Your Setup

### For Teams
1. Share this repository link
2. Document your model preferences  
3. Create test audio dataset
4. Share performance benchmarks

### For Presentations
1. Prepare 3-5 diverse audio files
2. Test search queries beforehand
3. Enable metrics dashboard
4. Have backup queries ready

---

## Need Help?

**Community Support:**
- GitHub Issues: Report bugs or ask questions
- GitHub Discussions: Share use cases and tips

**Documentation:**
- Main README: Project overview and features
- This Guide: Setup and deployment help
- Code Comments: Technical implementation details

**Quick Testing:**
Try these search queries with music files:
- `"drums and bass"`
- `"piano melody"`  
- `"electronic music"`
- `"acoustic guitar"`
- `"vocal singing"`