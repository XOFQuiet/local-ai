import os
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

# Enable secure cross-origin requests so your UI dashboard can communicate seamlessly
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
# Your local independent Docker SearXNG gateway
SEARXNG_URL = "http://localhost:8080/search"

# Creating a secure sandbox workspace folder on your machine
WORKSPACE_DIR = os.path.join(os.getcwd(), "workspace_files")
if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR)

# Constrained, analytical system instructions preventing narrative drift and hallucinations
SYSTEM_PROMPT = """You are an advanced, literal-minded AI Agent workspace manager with private search access.
Your primary directive is to execute operations using exact JSON payloads, ensuring 100% accurate spelling.

CRITICAL PROCESSING RULES:
1. You can only output ONE JSON block at a time. Do NOT chain actions together.
2. Hard Fact-Checking: When analyzing text inside the <verified_web_data> tags, you are strictly forbidden from guessing, exaggerating, or making up numerical data (such as total retirements, point gaps, or lap times).
3. If the provided data contains conflicting statements, state ONLY what is explicitly agreed upon across sources. If specific stats (like exact DNF counts) are not clearly summarized, omit them entirely.

To search the web:
{"action": "search", "query": "optimized search terms here"}

To write a file:
{"action": "write", "filename": "filename.txt", "content": "Exact factual text based ONLY on the XML tags"}

To read a file:
{"action": "read", "filename": "filename.txt"}

To list files:
{"action": "list"}"""

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {request.message}\nAgent:"
        return await call_ollama_and_parse(full_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def call_ollama_and_parse(prompt_text: str):
    payload = {
        "model": "qwen2.5:7b",
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "temperature": 0.0,  # Destroys random creativity
            "top_p": 0.1,        # Restricts token selection to the most mathematically certain words
            "num_predict": 256   # Keeps responses short, crisp, and dense with facts
        }
    }
    
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    ai_text = response.json().get("response", "").strip()
    
    # Strip potential markdown code wrapper syntax blocks if returned by the model
    if "```json" in ai_text:
        ai_text = ai_text.split("```json")[1].split("```")[0].strip()
    elif "```" in ai_text:
        ai_text = ai_text.split("```")[1].split("```")[0].strip()

    try:
        tool_call = json.loads(ai_text)
        action = tool_call.get("action")
        
        # --- SEARXNG MULTI-TURN SEARCH PIPELINE ---
        if action == "search":
            query = tool_call["query"]
            print(f"--- AGENT ACTION: Querying local SearXNG cluster with optimized query: '{query}' ---")
            
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                params = {"q": query, "format": "json"}
                
                search_response = requests.get(SEARXNG_URL, params=params, headers=headers, timeout=10)
                search_response.raise_for_status()
                
                search_data = search_response.json()
                results = search_data.get("results", [])[:4]
                
                # Format our gathered stream using rigid context boundaries for smaller models
                search_summary = "<verified_web_data>\n"
                search_summary += f"Current Date Context: {datetime.now().strftime('%A, %B %d, %Y')}\n\n"
                
                for idx, r in enumerate(results):
                    search_summary += f"<source id='{idx+1}'>\n"
                    search_summary += f"  <title>{r.get('title', '')}</title>\n"
                    search_summary += f"  <snippet>{r.get('content', '')}</snippet>\n"
                    search_summary += f"</source>\n"
                search_summary += "</verified_web_data>"
                
            except Exception as search_err:
                print(f"--- SEARXNG ERROR: {search_err} ---")
                search_summary = f"<error>Could not sync with local cluster: {str(search_err)}</error>"
            
            # Re-inject the pristine data block back into the model context window for final execution
            follow_up_prompt = f"{prompt_text}\n{ai_text}\n\nSystem: Here is the verified search data context:\n{search_summary}\n\nSynthesize this data perfectly. Write your final answer or file action JSON based strictly on the factual details provided inside the XML tags."
            return await call_ollama_and_parse(follow_up_prompt)

        elif action == "write":
            file_path = os.path.join(WORKSPACE_DIR, tool_call["filename"])
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(tool_call["content"])
            return {"response": f"📁 Action Completed: I successfully wrote your notes to '{tool_call['filename']}' inside your workspace folder."}
            
        elif action == "read":
            file_path = os.path.join(WORKSPACE_DIR, tool_call["filename"])
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    return {"response": f"📖 Content of '{tool_call['filename']}':\n\n{f.read()}"}
            return {"response": f"❌ Error: File '{tool_call['filename']}' does not exist."}
            
        elif action == "list":
            files = os.listdir(WORKSPACE_DIR)
            return {"response": json.dumps(files)}
            
    except json.JSONDecodeError:
        # Graceful fallback to conversational UI display if text output occurs
        return {"response": ai_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)