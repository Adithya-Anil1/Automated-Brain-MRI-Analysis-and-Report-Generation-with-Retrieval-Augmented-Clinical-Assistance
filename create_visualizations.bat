@echo off
cd /d "c:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant"
venv310\Scripts\python.exe scripts\visualize_segmentation.py --mri "data\temp_inference_input" --seg "results\test_run\BraTS2021_00495.nii.gz" --output "visualizations" --slices 9
echo.
echo Visualizations created in the 'visualizations' folder!
pause
