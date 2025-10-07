{
  "project": {
    "name": "AI-Powered Brain MRI Assistant",
    "goal": "Automate tumor segmentation, report generation, and therapy prediction from multimodal MRI scans",
    "tech_stack": ["Python", "PyTorch", "TensorFlow", "Flask", "React"],
    "modules": {
      "segmentation": "3D U-Net + Vision Transformers for tumor masks",
      "classification": "CNNs + XGBoost for tumor type detection (gliomas, meningiomas, metastases)",
      "report_generation": "NLP (BioBERT/GPT) to produce structured clinical reports",
      "therapy_prediction": "CNN-LSTM/RNN for longitudinal treatment response"
    },
    "inputs": ["T1", "T1Gd", "T2", "FLAIR MRI sequences"]
  },
  "guidelines": {
    "file_creation": "You may create new files if needed, but do not create multiple or duplicate files unnecessarily. Only create files explicitly requested or clearly required.",
    "coding": "Keep code modular, well-commented, follow naming conventions, avoid unnecessary boilerplate",
    "preprocessing": "Normalize, resize, augment MRI scans",
    "outputs": "Generate readable, structured reports (JSON, Markdown, or tables). No need for strict clinical standards."
  },
  "libraries": ["PyTorch", "TensorFlow", "HuggingFace", "Nibabel"]
}
