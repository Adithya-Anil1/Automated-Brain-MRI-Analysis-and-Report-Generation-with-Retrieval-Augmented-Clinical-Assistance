
## 🎯 Project Overview

This is an **AI-assisted diagnostic system** for brain tumor segmentation using pretrained nnU-Net models. The system analyzes multi-modal MRI scans (T1, T1CE, T2, FLAIR) and automatically identifies tumor regions to assist radiologists in diagnosis.

### What This MVP Does:
✅ Loads a **pretrained nnU-Net model** (no training required)  
✅ Segments brain tumors into 3 regions: NCR, ED, ET  
✅ Calculates tumor volumes automatically  
✅ Generates **visual overlays** showing segmentation results  
✅ Provides structured output for integration into reports  

### Tumor Segmentation Labels:
- **NCR (Label 1)**: Necrotic/Non-enhancing Tumor Core
- **ED (Label 2)**: Peritumoral Edema
- **ET (Label 3)**: Enhancing Tumor

### Clinical Metrics:
- **Whole Tumor (WT)**: NCR + ED + ET (all tumor regions)
- **Tumor Core (TC)**: NCR + ET (solid tumor parts)
- **Enhancing Tumor (ET)**: Active tumor region

---

## 📁 Project Files

```
AI-Powered Brain MRI Assistant/
├── requirements.txt              # Python dependencies
├── setup_nnunet.py              # Creates directory structure
├── download_pretrained_model.py # Downloads pretrained nnU-Net
├── inference_nnunet.py          # Runs segmentation & visualization
├── validate_setup.py            # Checks if setup is correct
└── README.md                    # This file
```


## 📊 Understanding the Output

### **1. Segmentation Mask** (`segmentation_result.nii.gz`)
- 3D NIfTI file with the same dimensions as input
- Each voxel labeled: 0 (background), 1 (NCR), 2 (ED), 3 (ET)
- Can be loaded in medical imaging software (3D Slicer, ITK-SNAP)

### **2. Visualization** (`visualization.png`)
Three panels:
- **Left:** Original T1CE MRI slice
- **Middle:** Segmentation mask (color-coded)
- **Right:** Overlay (mask on top of MRI)

Color coding:
- 🔵 Blue = NCR (Necrotic core)
- 🟢 Green = ED (Edema)
- 🔴 Red = ET (Enhancing tumor)

### **3. Volume Measurements**
Printed to console and can be saved to file for report generation

---

### **Integration Ideas:**

1. **Automated Report Generation:**
   - Parse the volume output from `inference_nnunet.py`
   - Generate structured reports using templates
   - Include visualizations in PDF reports

2. **Batch Processing:**
   - Modify script to process multiple patients
   - Store results in database
   - Generate comparison reports

3. **Quality Control:**
   - Visual review of segmentations
   - Flag cases with unusual volumes
   - Export for radiologist confirmation

---

## 📚 References

- **nnU-Net Paper:** [Nature Methods 2021](https://www.nature.com/articles/s41592-020-01008-z)
- **nnU-Net GitHub:** https://github.com/MIC-DKFZ/nnUNet
- **BraTS Challenge:** https://www.synapse.org/brats

---



