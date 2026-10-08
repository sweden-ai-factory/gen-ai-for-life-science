# Setup

This workshop will use **JupyterLab via Open OnDemand on LUMI** as the primary interface for all practical sessions. Google Colab will be available as a fallback option if needed.


## Open OnDemand Setup

### Accessing JupyterLab

1. Open the provided Open OnDemand link in your browser
2. Log in with the credentials provided by workshop organizers
3. Navigate to the JupyterLab launch option
4. Start a new JupyterLab session with GPU support


## Google Colab Fallback Setup

If needed, you can use Google Colab as an alternative:

### Accessing Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in with your Google account
3. Enable GPU acceleration:
   - Go to **Runtime** > **Change runtime type**
   - Select **GPU** as the hardware accelerator, if available
   - Click **Save**

## Checking Your Environment

Once in, you can verify your setup by running:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("PyTorch version:", torch.__version__)
```

## Datasets and Tools used in the workshop

### Protein Structure Prediction

For Session 3, we'll use online tools:
- [ESMFold](https://esmatlas.com/resources?action=fold) - For protein structure prediction
- [ColabFold](https://github.com/sokrypton/ColabFold) - Alternative protein folding tool

### Medical Imaging Datasets

For Session 4, we'll work with:
- [IU X-ray dataset](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) - Chest X-rays with radiology reports via [OpenI](https://openi.nlm.nih.gov/)
