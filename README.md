## Python RAG Setup (Google Gemini Powered)

DocPilot AI backend uses Google Gemini (`models/text-embedding-004` and `gemini-1.5-flash`) for fast, lightweight, and high-accuracy document intelligence with zero heavy PyTorch dependencies.

### Local Setup

From the `backend` directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file with your free Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey))*

Run verification:
```powershell
python verify_setup.py
```

Start the backend server:
```powershell
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
