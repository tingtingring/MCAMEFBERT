# MCAMEFBERT

MCAMEFBERT is a DNA sequence classification model that incorporates the [DNABERT-2-117M](https://huggingface.co/zhihan1996/DNABERT-2-117M) pre-trained model as part of its architecture.  
Specifically, DNABERT-2 is used within MCAMEFBERT to extract rich sequence embeddings, which are further processed by the model's custom multi-channel attention and other task-specific modules.

---

🧬 Model Configuration

- **Pre-trained model dependency**:  
  MCAMEFBERT internally utilizes the DNABERT-2 model ([zhihan1996/DNABERT-2-117M](https://huggingface.co/zhihan1996/DNABERT-2-117M)) for sequence representation learning.  
  Please ensure the model can be downloaded from Hugging Face Hub or pre-downloaded to a local directory.

- **Sequence length**:  
  The model uses a fixed input sequence length of `201`.

- **Directories for outputs**:  
  - `save/seq_len201/` - for saving model checkpoints and intermediate files.
  - `results/seq_len201/` - for storing evaluation results and log files.

> ⚠️ Make sure the directories `save/seq_len201` and `results/seq_len201` are created before running the code to avoid path errors.

---

📦 Environment Setup

Install the required Python packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt

📁 Project Structure
MCAMEFBERT/
├── save/
│   └── seq_len201/        # Model checkpoints and intermediate files
├── results/
│   └── seq_len201/        # Evaluation results and logs
├── requirements.txt       # Environment configuration file
├── train.py               # Training script
└── evaluate.py            # Evaluation script

🚀 Quick Start
1. Download the pre-trained model
Make sure the DNABERT-2 pre-trained model is accessible:

https://huggingface.co/zhihan1996/DNABERT-2-117M

You can let Hugging Face Transformers automatically download the model during the first run or manually download it in advance.

2. Install the environment
```bash
pip install -r requirements.txt

3. Train the model
```bash
python train.py --config config.yaml

💡 Notes
Make sure your data is prepared properly (see data format and paths in your configuration file).

The model is designed to run on GPU by default. To run on CPU, adjust the configurations accordingly.

Ensure your network and Hugging Face authentication tokens (if required) are correctly set up when downloading models from the Hugging Face Hub.
