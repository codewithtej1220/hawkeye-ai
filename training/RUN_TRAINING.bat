@echo off
title HawkEye - Train YOLO11m on real thermal + RGB data
cd /d "%~dp0"

echo [1/4] Downloading real datasets from Kaggle...
python download_datasets.py || goto :error

echo [2/4] Building merged person/weapon dataset...
python prepare_dataset.py --delete-raw || goto :error

echo [3/4] Training YOLO11m (several hours on a laptop GPU)...
python train.py || goto :error

echo [4/4] Per-domain evaluation...
python evaluate.py || goto :error

echo.
echo Done. Model saved to project\models\hawkeye_real_yolo11m.pt
echo Restart the backend to load it.
pause
exit /b 0

:error
echo.
echo Training pipeline failed - see output above.
pause
exit /b 1
