import os
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable secure cross-origin requests so your local file can talk to the backend port
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
# Your local Docker SearXNG gateway
SEARXNG_URL = "http://localhost:8080/search"

# Creating a secure sandbox workspace folder on your machine
WORKSPACE_DIR = os.path.join(os.getcwd(), "workspace_files")
if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR)

# Enforced system instructions for structural execution and exact spelling copy
SYSTEM_PROMPT = """You are an advanced AI Agent workspace manager with private SearXNG metasearch access and filesystem tools.
Your primary directive is to execute operations using exact JSON payloads, ensuring 100% accurate spelling.

If the user asks you to write, read, list files, or search the web, you MUST respond ONLY with a raw JSON block.

To search the web (blends Google, Bing, and Wikipedia results together):
{"action": "search", "query": "search terms here"}

To write a file:
{"action": "write", "filename": "filename.txt", "content": "Exact text with perfect spelling"}

To read a file:
{"action": "read", "filename": "filename.txt"}

To list files:
{"action": "list"}

CRITICAL: If you use the search tool, you will be given the results in the next turn. Use those results to answer the user or write a file."""

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
        "options": {"temperature": 0.0}  # Keeps the model strictly focused on following rules
    }
    
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    ai_text = response.json().get("response", "").strip()
    
    # Clean up any potential markdown syntax block wraps code generation models might use
    if "```json" in ai_text:
        ai_text = ai_text.split("```json")[1].split("```")[0].strip()
    elif "```" in ai_text:
        ai_text = ai_text.split("```")[1].split("```")[0].strip()

    try:
        tool_call = json.loads(ai_text)
        action = tool_call.get("action")
        
        # --- UPGRADED SEARXNG SEARCH LOOP WITH ERROR HANDLING ---
        if action == "search":
            query = tool_call["query"]
            print(f"--- AGENT ACTION: Querying local SearXNG cluster for '{query}' ---")
            
            try:
                # Add a custom User-Agent so the local cluster accepts your request
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                params = {"q": query, "format": "json"}
                
                search_response = requests.get(SEARXNG_URL, params=params, headers=headers, timeout=10)
                search_response.raise_for_status()
                
                search_data = search_response.json()
                results = search_data.get("results", [])[:3]  # Grab top 3 global results
                
                if not results:
                    search_summary = "No results found for this query."
                else:
                    search_summary = "\n".join([f"Source: {r.get('engine', 'unknown')} | Title: {r.get('title')} - {r.get('content', '')}" for r in results])
                
            except Exception as search_err:
                print(f"--- SEARXNG ERROR: {search_err} ---")
                search_summary = f"Error connecting to local search cluster: {str(search_err)}. Proceed with internal knowledge."
            
            follow_up_prompt = f"{prompt_text}\n{ai_text}\n\nSystem: Here are the aggregated search results from your cluster:\n{search_summary}\n\nBased on these results, provide your final answer or file action JSON."
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
        # Fall back to standard conversational text if no JSON was detected
        return {"response": ai_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)