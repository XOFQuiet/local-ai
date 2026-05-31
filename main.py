import os
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable secure communication between your frontend webpage and backend server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

OLLAMA_URL = "http://localhost:11434/api/generate"

# Set up the secure sandbox folder for the AI to interact with
WORKSPACE_DIR = os.path.join(os.getcwd(), "workspace_files")
if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR)

# Enforced system instructions to maintain both JSON structure and text spelling accuracy
SYSTEM_PROMPT = """You are a highly precise local AI Agent workspace manager. 
Your primary directive is to execute filesystem operations using exact JSON payloads, while ensuring all user content is copied with 100% accurate spelling, grammar, and formatting.

If the user asks you to write, read, or list files, you MUST respond ONLY with a raw JSON block. Do not misspell words inside the content block.

To write a file:
{"action": "write", "filename": "filename.txt", "content": "Exact user text with perfect spelling"}

To read a file:
{"action": "read", "filename": "filename.txt"}

To list files:
{"action": "list"}

CRITICAL: Double-check your spelling before closing the JSON string. If the user says 'eggs', do not write 'eygs'."""

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {request.message}\nAgent:"
        
        payload = {
            "model": "qwen2.5:7b",
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0  # Keeps the model strictly focused on following rules
            }
        }
        
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        ai_text = response.json().get("response", "").strip()
        
        # Strip markdown syntax wraps if the AI adds them unexpectedly
        if "```json" in ai_text:
            ai_text = ai_text.split("```json")[1].split("```")[0].strip()
        elif "```" in ai_text:
            ai_text = ai_text.split("```")[1].split("```")[0].strip()

        # Try to execute the tool call requested by the AI
        try:
            tool_call = json.loads(ai_text)
            action = tool_call.get("action")
            
            if action == "write":
                file_path = os.path.join(WORKSPACE_DIR, tool_call["filename"])
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(tool_call["content"])
                return {"response": f"📁 Action Completed: I successfully wrote your notes to '{tool_call['filename']}' inside your workspace folder."}
                
            elif action == "read":
                file_path = os.path.join(WORKSPACE_DIR, tool_call["filename"])
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    return {"response": f"📖 Content of '{tool_call['filename']}':\n\n{content}"}
                return {"response": f"❌ Error: File '{tool_call['filename']}' does not exist."}
                
            elif action == "list":
                files = os.listdir(WORKSPACE_DIR)
                if not files:
                    return {"response": "📁 The workspace folder is currently empty."}
                return {"response": f"📁 Files in workspace:\n" + "\n".join([f"- {file}" for file in files])}
                
        except json.JSONDecodeError:
            # Fallback to normal text if it's a standard chat message
            return {"response": ai_text}
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama connection error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)