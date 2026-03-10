# Deepfake AI Image and Text Detection System

A comprehensive full-stack AI-powered system for detecting deepfake images and AI-generated text using state-of-the-art deep learning models.

![Deepfake Detection](https://img.shields.io/badge/Deepfake-Detection-blue)
![AI](https://img.shields.io/badge/AI-Powered-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-lightgrey)
![React](https://img.shields.io/badge/Frontend-HTML%20%2B%20Tailwind-orange)

## 🌟 Features

### Image Deepfake Detection
- **ResNet50-based CNN model** for binary classification (Real vs Fake)
- **Transfer learning** with pretrained ImageNet weights
- **Grad-CAM heatmap visualization** showing manipulated regions
- Support for multiple image formats (JPG, PNG, WebP)
- Confidence score percentage

### AI Text Detection
- **BERT-base-uncased model** from HuggingFace Transformers
- Contextual understanding of text patterns
- Binary classification (Human vs AI-generated)
- Handles texts up to 5000 characters
- Precision and confidence metrics

### User Interface
- Modern, responsive design with Tailwind CSS
- Drag-and-drop image upload
- Real-time detection with loading animations
- Visual confidence bars and color-coded results
- Heatmap toggle for manipulated region visualization
- Sample text examples for testing

### Backend API
- FastAPI framework for high performance
- RESTful endpoints for image and text detection
- CORS-enabled for frontend integration
- Health check and model info endpoints
- Error handling and validation

## 📁 Project Structure

```
deepfake-detector/
├── backend/
│   ├── app.py                 # FastAPI application with API endpoints
│   ├── image_model.py         # ResNet50 image detection model
│   ├── text_model.py          # BERT text detection model
│   ├── utils.py               # Helper functions (heatmap, preprocessing)
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── index.html             # Main UI with Tailwind CSS
│   └── script.js              # Frontend JavaScript for API calls
├── models/
│   ├── image_model.pth        # Trained image detection weights
│   └── text_model/            # Trained BERT model directory
├── data/
│   ├── images/                # Uploaded images storage
│   ├── sample_data/           # Sample test images
│   └── training/              # Training dataset directory
├── train_image_model.py       # Image model training script
├── train_text_model.py        # Text model training script
├── test_demo.py               # Demo and testing script
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)
- Modern web browser (Chrome, Firefox, Edge)

### Installation

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd deepfake-detector
```

#### 2. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Note:** The first installation may take several minutes as it downloads PyTorch, TensorFlow, and other large packages.

#### 3. Download or Train Models

The system requires trained models to function. You have two options:

**Option A: Use Pretrained Models (Recommended)**

Download pretrained model weights and place them in the `models/` directory:
- `image_model.pth` - Place in `models/`
- `text_model/` - Place entire directory in `models/`

**Option B: Train Your Own Models**

See the [Training Section](#training-models-from-scratch) below.

#### 4. Run the Backend Server

```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

#### 5. Open the Frontend

Simply open `frontend/index.html` in your web browser, or use a local server:

```bash
# Option 1: Python HTTP server
cd frontend
python -m http.server 3000

# Option 2: Using live-server (if installed)
live-server frontend
```

Navigate to `http://localhost:3000`

## 📖 Usage

### Image Detection

1. Click "Select Image" or drag and drop an image
2. Wait for the preview to load
3. Click "Analyze Image"
4. View results:
   - **Prediction**: Real or Fake
   - **Confidence**: Percentage score
   - **Heatmap**: Shows manipulated regions (red areas indicate manipulation)

### Text Detection

1. Enter or paste text in the input area (minimum 10 characters)
2. Optionally use sample text buttons for testing
3. Click "Analyze Text"
4. View results:
   - **Prediction**: Human or AI Generated
   - **Confidence**: Percentage score

### Testing with Demo Script

Use the included demo script to test models without the web interface:

```bash
python test_demo.py
```

This will:
- Check model status
- Test with sample images
- Test with sample texts
- Display detailed predictions

## 🎯 API Endpoints

### Base URL
`http://localhost:8000`

### Available Endpoints

#### GET `/`
Returns API information and available endpoints.

#### GET `/health`
Health check endpoint with model status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "models": {
    "image_model_loaded": true,
    "text_model_loaded": true,
    "cuda_available": false
  }
}
```

#### POST `/detect-image`
Detect whether an image is real or fake.

**Request:**
- Content-Type: `multipart/form-data`
- File: Image file (JPG, PNG, WebP)

**Response:**
```json
{
  "success": true,
  "prediction": "Fake",
  "confidence": 94.56,
  "confidence_percentage": "94.56%",
  "color": "#EF4444",
  "filename": "uuid-here.jpg",
  "timestamp": "2024-01-01T12:00:00"
}
```

#### POST `/detect-image-with-heatmap`
Same as above but includes Grad-CAM heatmap.

**Response:**
```json
{
  "success": true,
  "prediction": "Fake",
  "confidence": 94.56,
  "heatmap_url": "/static/heatmaps/heatmap_uuid.jpg",
  "heatmap_base64": "data:image/jpeg;base64,...",
  ...
}
```

#### POST `/detect-text`
Detect whether text is human-written or AI-generated.

**Request:**
- Content-Type: `application/json`
- Body: `{"text": "Your text here..."}`

**Response:**
```json
{
  "success": true,
  "prediction": "AI Generated",
  "confidence": 87.23,
  "confidence_percentage": "87.23%",
  "color": "#EF4444",
  "text_length": 245,
  "timestamp": "2024-01-01T12:00:00"
}
```

#### GET `/model-info`
Get detailed information about loaded models.

## 🎓 Training Models from Scratch

### Image Model Training

#### 1. Prepare Dataset

Organize your training data:

```
data/training/images/
├── real/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── fake/
    ├── deepfake1.jpg
    ├── deepfake2.png
    └── ...
```

**Recommended Datasets:**
- FaceForensics++
- DeepFake Detection Challenge (DFDC)
- Celeb-DF
- FF++

#### 2. Run Training

```bash
python train_image_model.py
```

**Training Parameters:**
- Batch size: 32
- Epochs: 15
- Learning rate: 0.001
- Image size: 224x224
- Model: ResNet50 with transfer learning

**Output:**
- `models/image_model.pth` - Best model weights
- `training_plot.png` - Loss and accuracy curves

### Text Model Training

#### 1. Prepare Dataset

**Option A: Directory Structure**

```
data/training/text/
├── human/
│   ├── text1.txt
│   ├── text2.txt
│   └── ...
└── ai/
    ├── generated1.txt
    ├── generated2.txt
    └── ...
```

**Option B: JSON File**

Create a JSON file with labeled data:
```json
[
  {"text": "Human written text...", "label": 0},
  {"text": "AI generated text...", "label": 1}
]
```

**Recommended Datasets:**
- HC3 (Human ChatGPT Comparison Corpus)
- OpenAI Text Classifier dataset
- Custom collected samples

#### 2. Run Training

```bash
# Using directory structure
python train_text_model.py --data-dir data/training/text

# Using JSON file
python train_text_model.py --json-file data/dataset.json

# Custom parameters
python train_text_model.py --epochs 5 --batch-size 16 --lr 2e-5
```

**Training Parameters:**
- Batch size: 16
- Epochs: 5
- Learning rate: 2e-5
- Max length: 512 tokens
- Model: BERT-base-uncased

**Output:**
- `models/text_model/` - Complete model directory
- `text_training_plot.png` - Training curves
- `training_metadata.json` - Training metrics

## 🔧 Configuration

### Environment Variables (Optional)

Create a `.env` file in the `backend/` directory:

```env
# Model paths
IMAGE_MODEL_PATH=models/image_model.pth
TEXT_MODEL_PATH=models/text_model

# Server configuration
HOST=0.0.0.0
PORT=8000

# Upload limits
MAX_FILE_SIZE_MB=10
MAX_TEXT_LENGTH=5000
```

### Modifying Model Architecture

You can modify the model architectures in:
- `backend/image_model.py` - Change ResNet layers, dropout rates
- `backend/text_model.py` - Adjust BERT layers, classifier head

## 🐛 Troubleshooting

### Backend Won't Start

**Issue:** ModuleNotFoundError

**Solution:**
```bash
cd backend
pip install -r requirements.txt --upgrade
```

### CUDA Not Available

**Issue:** Running on CPU only

**Solution:**
1. Install CUDA-enabled PyTorch:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
2. Ensure NVIDIA drivers are installed

### Model Not Loading

**Issue:** "Using randomly initialized weights"

**Solution:**
- Verify model files exist in `models/` directory
- Check file permissions
- Re-run training scripts

### Frontend Can't Connect

**Issue:** "Backend offline" message

**Solution:**
1. Ensure backend is running: `http://localhost:8000/health`
2. Check CORS settings in `backend/app.py`
3. Verify no firewall blocking port 8000

### Low Accuracy

**Issue:** Poor detection performance

**Solution:**
- Train with more diverse dataset
- Increase training epochs
- Use data augmentation
- Fine-tune hyperparameters

## 📊 Performance Metrics

### Image Model (Expected)

With proper training on standard datasets:
- **Accuracy**: 85-95%
- **Precision**: 88-96%
- **Recall**: 82-94%
- **Inference Time**: 0.1-0.5 seconds per image

### Text Model (Expected)

With proper training on HC3 or similar:
- **Accuracy**: 80-92%
- **Precision**: 78-90%
- **Recall**: 75-88%
- **Inference Time**: 0.05-0.2 seconds per text

## 🔒 Security Considerations

- **File Upload Validation**: All uploads are validated for type and size
- **Input Sanitization**: Text inputs are sanitized before processing
- **CORS Configuration**: Update `allow_origins` in production
- **Rate Limiting**: Consider adding rate limiting for production use

## 🚀 Deployment

### Production Server

Use a production ASGI server:

```bash
# With Gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# With Uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t deepfake-detector .
docker run -p 8000:8000 deepfake-detector
```

## 📚 Additional Resources

### Research Papers
- [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1906.00804)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)

### Datasets
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [DFDC Dataset](https://ai.facebook.com/datasets/dfdc/)
- [HC3 Dataset](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)

### Tools & Libraries
- [PyTorch](https://pytorch.org/)
- [HuggingFace Transformers](https://huggingface.co/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open-source and available under the MIT License.

## 👥 Authors

Built with ❤️ using FastAPI, PyTorch, and BERT

## 🙏 Acknowledgments

- FaceForensics++ team for dataset
- HuggingFace for Transformers library
- FastAPI community
- PyTorch team

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section

---

**Made with cutting-edge AI technology for detecting deepfakes and AI-generated content.**
