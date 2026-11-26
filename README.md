# DeepFER Phase 4 — Streamlit + Docker Deployment

## Included
- `app.py`: Streamlit app with image upload and camera capture.
- `models/deepfer_best_model.keras`: Phase 3 selected MobileNetV2 model.
- `models/model_metadata.json`: class names and model input metadata.
- `reports/`: Phase 3 validation/test summary artifacts.
- `Dockerfile` and `docker-compose.yml`.

## 1. Run locally (Windows PowerShell)

Open PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Run with Docker Desktop

Start Docker Desktop and wait until its engine is running. From this folder:

```powershell
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f deepfer
```

Open `http://localhost:8501`.

Stop:
```powershell
docker compose down
```

Rebuild after code/model changes:
```powershell
docker compose down
docker compose build --no-cache
docker compose up -d
```

## 3. Test the application
1. Upload a clear, front-facing face image, or use the camera input.
2. Check the face crop preview.
3. Click **Classify expression**.
4. Review the top category and class scores.
5. Try different lighting, pose and image quality conditions.

## 4. Important model preprocessing
Phase 3 selected **MobileNetV2 fine-tuned**. Training transformed 48×48 grayscale images to 3-channel RGB and resized them to 96×96. The model itself includes the rescaling layer to [-1, 1]. The app repeats the grayscale-to-RGB and 96×96 resize steps; do not apply MobileNetV2 `preprocess_input` a second time.

The app uses OpenCV Haar face detection to crop the largest detected face. If no face is found, it warns and uses the full image. Haar detection is a simple baseline and may fail for side poses, occlusion, or poor lighting.

## 5. Phase 3 performance snapshot
- Validation: MobileNetV2 accuracy 56.06%, macro F1 0.553.
- Test: accuracy 55.47%, macro F1 0.532.
- Stronger test classes: happy (F1 0.751), surprise (F1 0.675).
- Weaker test class: fear (F1 0.374).

This is a modest research baseline, not a production-grade emotion detector. Treat outputs as visible-expression labels only, not facts about internal emotional states. Do not use for consequential decisions about people. Obtain consent before capturing or processing anyone's face and avoid storing uploaded images.

## Troubleshooting
- **TensorFlow install issue:** use Python 3.11 in the local environment; Docker already uses Python 3.11.
- **Port 8501 in use:** change the left side of the compose port mapping, e.g. `8502:8501`, then open port 8502.
- **Docker daemon unavailable:** start Docker Desktop, wait for it to finish starting, then retry.
- **Model file missing:** confirm `models/deepfer_best_model.keras` exists and is included in the project folder.
