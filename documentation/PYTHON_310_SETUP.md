# Python 3.10 Setup Guide for AI-Powered Brain MRI Assistant

## Step 1: Download Python 3.10

1. **Visit:** https://www.python.org/downloads/release/python-31011/
2. **Scroll down** to "Files" section
3. **Download:** `Windows installer (64-bit)` 
   - File: `python-3.10.11-amd64.exe` (~27 MB)

## Step 2: Install Python 3.10

1. **Run** the installer
2. **IMPORTANT:** Check these boxes:
   - ☑ "Add python.exe to PATH"
   - ☑ "Install launcher for all users (recommended)"
3. Click **"Customize installation"**
4. **Optional Features** - check all (default is fine)
5. **Advanced Options:**
   - ☑ Install for all users
   - ☑ Add Python to environment variables
   - Install location: `C:\Python310\` (or default is fine)
6. Click **Install**

## Step 3: Verify Installation

Open a **NEW** Command Prompt and run:
```cmd
py -3.10 --version
```

You should see: `Python 3.10.11`

## Step 4: Create Virtual Environment for This Project

Navigate to your project folder and create a virtual environment:

```cmd
cd "c:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant"
py -3.10 -m venv venv310
```

This creates a `venv310` folder with Python 3.10.

## Step 5: Activate the Virtual Environment

```cmd
venv310\Scripts\activate
```

Your prompt should change to show `(venv310)` at the beginning.

## Step 6: Install All Packages

With the virtual environment activated, install everything:

```cmd
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install nnunetv2
pip install SimpleITK nibabel scipy scikit-image matplotlib
pip install medpy==0.4.0 batchgenerators axial-attention==0.5.0 monai
```

This will take 5-10 minutes (downloading ~3 GB).

## Step 7: Verify CUDA Support

```cmd
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Should show: `CUDA: True, GPU: NVIDIA GeForce RTX 4050 Laptop GPU`

## Step 8: Run Segmentation

Now you can run the segmentation with proper multiprocessing support:

```cmd
python run_brats2021_inference.py --input sample_data\BraTS2021_sample --output results\BraTS2021_00495
```

---

## Future Use

**Every time** you want to work on this project:

1. Open Command Prompt
2. Navigate to project folder:
   ```cmd
   cd "c:\Users\adith\OneDrive\Desktop\AI-Powered Brain MRI Assistant"
   ```
3. Activate virtual environment:
   ```cmd
   venv310\Scripts\activate
   ```
4. Run your scripts!

To deactivate when done:
```cmd
deactivate
```

---

## Notes

- ✅ Python 3.12 stays untouched - still available for other projects
- ✅ Virtual environment is isolated - won't affect system Python
- ✅ Can delete `venv310` folder anytime to start fresh
- ✅ Add `venv310/` to `.gitignore` (already done)

---

## Quick Commands Reference

| Task | Command |
|------|---------|
| Check Python 3.10 installed | `py -3.10 --version` |
| Create venv | `py -3.10 -m venv venv310` |
| Activate venv | `venv310\Scripts\activate` |
| Deactivate venv | `deactivate` |
| Install package | `pip install <package>` |
| Run segmentation | `python run_brats2021_inference.py --input sample_data\BraTS2021_sample --output results\BraTS2021_00495` |

---

**Ready to start? Let me know when you've completed Step 2 (Python 3.10 installation)!**
