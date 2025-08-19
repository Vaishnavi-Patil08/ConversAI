from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from tools import build_pdf_retriever_tool, get_tools_list
from langchain.agents import create_react_agent,AgentExecutor
import tempfile
import prompts 
import config
import db
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI 
import psutil
from fastapi.responses import JSONResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://convers-ai-frontend.vercel.app", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm =  ChatGoogleGenerativeAI(model=config.LLM_MODEL, api_key=config.GOOGLE_API_KEY) 
pdf_tool_cache = {}

base_tools = get_tools_list()

memory = ConversationBufferWindowMemory(
            k=10,  # Remember last 10 exchanges
            memory_key="chat_history",
            return_messages=True
        )
past_logs = db.fetch_logs()

for log in past_logs[-10:]:
    user_input = log["user_input"]
    agent_response = log["agent_response"]
    # Save each pair into memory context
    memory.save_context({"input": user_input}, {"output": agent_response})

base_agent = create_react_agent(
    tools=base_tools,
    llm=llm,
    prompt=prompts.prompt,
)

base_executor = AgentExecutor.from_agent_and_tools(
    agent=base_agent,
    tools=base_tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

@app.api_route("/health", methods=["GET","OPTIONS"])
async def health_check(request:Request):
    if request.method=="OPTIONS":
        return JSONResponse(status_code=200,content={"status":"ok"})
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    return {
        "status": "ok",
        "cpu_percent": cpu_percent,
        "memory_used_mb": round(memory.used / 1024 / 1024, 2),
        "memory_total_mb": round(memory.total / 1024 / 1024, 2),
    }


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), user_id: str = Form(...)):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_file.write(await file.read())
    tmp_file.close()

    pdf_tool = build_pdf_retriever_tool(tmp_file.name)
    pdf_tool_cache[user_id] = pdf_tool

    return {"msg": "PDF uploaded and tool created"}


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default")
    query = data.get("message")

    # tools = get_tools_list()
    if user_id in pdf_tool_cache:
        tools = base_tools + [pdf_tool_cache[user_id]]

        agent = create_react_agent(
            tools=tools,
            llm=llm,
            prompt=prompts.prompt, 
        )


        agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=tools,
                memory=memory,
                verbose=True,
                handle_parsing_errors=True,
                # max_iterations=3
            )

        result = agent_executor.invoke({"input": query})
    else:
        result = base_executor.invoke({"input": query})
    answer = result.get("output") or result.get("result") or str(result)
    db.log_conversation(query, answer)
    return {"response": answer}