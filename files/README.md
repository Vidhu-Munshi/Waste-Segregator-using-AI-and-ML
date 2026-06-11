# WasteVision AI — Backend

## Setup

```bash
cd backend
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /detect | Upload image → classify + detect |
| POST | /ocr | Upload image → extract text |
| GET | /history | Fetch detection history |
| GET | /report/pdf | Download PDF report |
| WS | /ws/webcam | Live webcam stream |

## Response format

```json
{
  "class": "plastic",
  "confidence": 0.94,
  "recyclable": true,
  "hazard_level": "low",
  "bboxes": [{"label": "bottle", "confidence": 0.87, "bbox": [10, 20, 120, 200]}]
}
```

## Connect Frontend

Paste contents of `frontend_patch.js` into a `<script>` tag at the bottom of  
`ai-waste-platform.html` (just before `</body>`).  
Set `API_BASE` to your server URL.

## Train

**Classifier (TF/Keras):**
```bash
python training/train_classifier.py --data datasets/waste --epochs 20
```
Dataset must be ImageFolder format: `datasets/waste/{class_name}/*.jpg`

**YOLOv8:**
```bash
python training/train_yolo.py --data datasets/waste_yolo/data.yaml --epochs 50
```

## Datasets (Kaggle)

Download and extract into `datasets/`:
- https://www.kaggle.com/datasets/phenomsg/waste-classification → `datasets/waste/`
- https://www.kaggle.com/code/kerneler/starter-e-waste-dataset-93b07fb8-a → `datasets/waste/e-waste/`
- https://www.kaggle.com/datasets/msharathgowda/ocr-dataset → `datasets/ocr/`

## Project Structure

```
backend/
├── main.py
├── requirements.txt
├── frontend_patch.js
├── routes/
│   ├── detect.py       POST /detect
│   ├── ocr.py          POST /ocr
│   ├── history.py      GET /history
│   ├── report.py       GET /report/pdf
│   └── webcam.py       WS  /ws/webcam
├── services/
│   ├── classifier.py   MobileNetV3 TF classifier
│   ├── detector.py     YOLOv8 detection
│   ├── ocr_engine.py   EasyOCR
│   └── report_generator.py  ReportLab PDF
├── database/
│   └── db.py           SQLite via aiosqlite
├── training/
│   ├── train_classifier.py
│   └── train_yolo.py
├── models/             trained weights go here
├── uploads/            saved images
└── datasets/           training data
```
