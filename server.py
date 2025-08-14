from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import main  
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_input = body.get("message")
    response_obj = main.agent_executor.invoke({"input": user_input})
    assistant_reply = response_obj.get("output") or response_obj.get("result") or str(response_obj)
    return {"response": assistant_reply}
