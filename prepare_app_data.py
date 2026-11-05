"""Copy the files the web app needs from model_output/ into app_data/.
Run AFTER:  python Scripts/aqi_model.py train --raw Dataset/city_day.csv
Usage:      python Scripts/prepare_app_data.py
"""
import shutil, sys
from pathlib import Path

SRC, DST = Path("model_output"), Path("app_data")
FILES = ["aqi_model.json", "cleaned_data.csv", "model_data.csv", "test_predictions.csv"]
DST.mkdir(exist_ok=True)
missing = [f for f in FILES if not (SRC / f).exists()]
if missing:
    sys.exit(f"Missing in {SRC}/: {missing}. Run the train command first.")
for f in FILES:
    shutil.copy(SRC / f, DST / f)
    print("copied", f)
print("Done. app_data/ is ready; run: streamlit run app.py")
