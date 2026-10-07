# WindGuard AI — Wind Turbine Damage Detection

A deploy-ready Flask + YOLOv5 web app for the trained wind-turbine inspection model.

## Damage classes
- LE Erosion
- PU Tape Damage
- Paint Peeloff

## Before deployment
Put your trained model at:

```text
models/best.pt
```

The app downloads the YOLOv5 source automatically on first model load, so you do **not** need to commit the full YOLOv5 repository.

## Local run

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Render deployment

1. Create a GitHub repository.
2. Upload all project files, including `models/best.pt`.
3. In Render choose **New → Web Service** and connect the repository.
4. Build command:

```text
pip install -r requirements.txt
```

5. Start command:

```text
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 180
```

6. Deploy. Render provides an `onrender.com` URL.

### Important
GitHub blocks files over 100 MB. If `best.pt` is over 100 MB, use Git LFS or object storage instead of committing the file directly.

Do not upload any Roboflow API key or other secret to GitHub.
