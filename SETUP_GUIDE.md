# Quick Setup Guide - Deepfake AI Detection System

## 🚀 5-Minute Setup (If you have pretrained models)

### Step 1: Install Dependencies (2 minutes)
```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Verify Models Exist
Make sure these files exist:
- `models/image_model.pth`
- `models/text_model/pytorch_model.bin`

If not, train them or download pretrained weights.

### Step 3: Start Backend (30 seconds)
```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 4: Open Frontend
Open `frontend/index.html` in your web browser.

**Done!** ✅ You can now detect deepfakes!

---

## 📚 Full Setup (Training from Scratch)

### Prerequisites
- Python3.8+
- pip
- 4GB+ RAM
- GPU recommended but not required

### Step 1: Install Dependencies
```bash
cd deepfake-detector
pip install -r backend/requirements.txt
```

### Step 2: Prepare Training Data

#### For Image Model:
Organize your dataset:
```
data/training/images/
├── real/      # Real images here
└── fake/      # Fake/deepfake images here
```

Download datasets from:
- FaceForensics++
- DFDC (DeepFake Detection Challenge)
- Celeb-DF

#### For Text Model:
Create directory structure:
```
data/training/text/
├── human/     # Human-written texts
└── ai/        # AI-generated texts
```

Or use JSON format:
```json
[
  {"text": "Human text...", "label": 0},
  {"text": "AI text...", "label": 1}
]
```

Datasets:
- HC3 (Human ChatGPT Comparison)
- OpenAI Text Classifier dataset

### Step 3: Train Models

#### Train Image Model (~30-60 minutes)
```bash
python train_image_model.py
```

This will:
- Load and augment training data
- Fine-tune ResNet50
- Save best model to `models/image_model.pth`
- Create training plot

#### Train Text Model (~15-30 minutes)
```bash
# Using directory structure
python train_text_model.py --data-dir data/training/text

# Using JSON file
python train_text_model.py --json-file data/dataset.json
```

This will:
- Tokenize texts with BERT tokenizer
- Fine-tune BERT model
- Save to `models/text_model/`
- Display metrics

### Step 4: Test Models
```bash
python test_demo.py
```

Select option3 to test both image and text detection.

### Step 5: Run Full System

Terminal 1 (Backend):
```bash
cd backend
uvicorn app:app --reload
```

Terminal 2 (Frontend - optional):
```bash
cd frontend
python-m http.server 3000
```

Open browser to `http://localhost:3000` or just open `frontend/index.html`.

---

## 🎯 Usage Examples

### Detect Image Deepfake

**Via Web Interface:**
1. Upload image
2. Click "Analyze Image"
3. View result + heatmap

**Via API:**
```bash
curl -X POST http://localhost:8000/detect-image \
  -F "file=@path/to/image.jpg"
```

### Detect AI Text

**Via Web Interface:**
1. Paste text
2. Click "Analyze Text"
3. View result

**Via API:**
```bash
curl -X POST http://localhost:8000/detect-text\
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here..."}'
```

---

## ⚠️ Common Issues

### Issue: ModuleNotFoundError
**Solution:**
```bash
pip install -r backend/requirements.txt --upgrade
```

### Issue: CUDA not available
**Solution:** Install GPU version of PyTorch:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Low accuracy
**Solution:**
- Train with more data
- Increase epochs
- Use data augmentation
- Fine-tune hyperparameters

### Issue: Backend won't start
**Check:**
```bash
cd backend
python -c "from app import app; print('OK')"
```

---

## 📖 Next Steps

1. **Test thoroughly** with known real/fake samples
2. **Fine-tune models** on your specific use case
3. **Deploy to production** using Docker or cloud service
4. **Add features** like user authentication, history, etc.

---

## 🔗 Resources

- **Full Documentation**: README.md
- **Project Summary**: PROJECT_SUMMARY.md
- **API Docs**: http://localhost:8000/docs (when backend running)

---

## 💡 Tips

1. **For best results**: Train on diverse, high-quality datasets
2. **GPU recommended**: Training is much faster with CUDA
3. **Start small**: Test with few samples first
4. **Monitor overfitting**: Watch validation accuracy
5. **Save checkpoints**: Training scripts auto-save best models

---

**Need Help?** Check the full README.md for detailed documentation.

Good luck detecting deepfakes! 🎯
