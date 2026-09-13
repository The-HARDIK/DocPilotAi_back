## Python RAG Setup

From the `backend` directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python verify_setup.py
```

The first successful run downloads the local embedding model
`BAAI/bge-small-en-v1.5` from Hugging Face. After that model is cached, you can
run verification without network checks:

```powershell
$env:HF_HUB_OFFLINE='1'
python verify_setup.py
```
