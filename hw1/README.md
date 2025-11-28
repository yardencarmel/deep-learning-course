## Homework 1 – Training LeNet-5 on Fashion-MNIST

This project trains several regularized variants of a LeNet-5–style CNN on the Fashion-MNIST dataset and logs convergence plots plus saved checkpoints. All scripts assume the current working directory is `hw1/`.

### 1. Prerequisites
- Python 3.11+ (tested with the version bundled in `hw1venv`)
- CUDA-capable GPU is optional; the code auto-detects and falls back to CPU


### 2. Recreate the Environment
```powershell
cd path\to\deep-learning-course\hw1
python -m venv hw1venv
hw1venv\Scripts\activate        # PowerShell / cmd
# source hw1venv/bin/activate   # macOS / Linux
pip install --upgrade pip
pip install -r requirements.txt # if present
# Otherwise install directly:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib tqdm
```
> If you already have `hw1venv`, run the appropriate activate command and skip the creation step.

### 3. Train All Models
```powershell
cd path\to\deep-learning-course\hw1
hw1venv\Scripts\activate
python main.py
```
The script:
- downloads Fashion-MNIST into `hw1/data` (once)
- trains Baseline, Dropout, Weight_Decay, and Batch_Normalization variants
- stores checkpoints in `hw1/models`
- saves convergence plots in `hw1/plots`
- writes final metrics to `hw1/results_summary.txt`

Training may take ~30–40 minutes on CPU

### 4. Test Saved Models
```powershell
cd path\to\deep-learning-course\hw1
hw1venv\Scripts\activate
python testing_models.py
```
This script loads each `{config}_{best|final}.pth` from `hw1/models`, evaluates on the test split (dropout disabled), and prints a summary table.

### 5. Troubleshooting
- **CUDA not available**: ensure GPU drivers + CUDA toolkit match your PyTorch build, or install the CPU wheel (`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`).
- **Dataset download errors**: delete `hw1/data/FashionMNIST` and re-run `main.py`; check network/firewall settings.
- **Matplotlib backend issues on headless Linux**: set `MPLBACKEND=Agg` before running `main.py`.

### 6. Expected Folder Layout (relative to `hw1/`)
```
hw1/
├─ main.py
├─ testing_models_3.py
├─ hw1venv/
├─ models/
├─ plots/
├─ data/
└─ results_summary.txt
```
Running everything from this directory ensures the path logic in the scripts resolves correctly on both Windows and Linux.

