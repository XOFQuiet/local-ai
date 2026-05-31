from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse  # <-- Added this
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# Allows your frontend webpage to talk to your backend server securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NEW: Serve the index.html file right from FastAPI ---
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>index.html file not found in directory!</h3>"
# --------------------------------------------------------

class ChatRequest(BaseModel):
    message: str

OLLAMA_URL = "http://localhost:11434/api/generate"

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        payload = {
            "model": "qwen2.5:7b",
            "prompt": request.message,
            "stream": False 
        }
        
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        
        ai_data = response.json()
        return {"response": ai_data.get("response", "No response received.")}
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama connection error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)