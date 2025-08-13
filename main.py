# main.py
import config
import db
from memory import VectorMemory
from tools import get_tools_list
from retriever import build_retriever_from_urls
# from prompts import chat_prompt
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.agents import initialize_agent,create_react_agent,AgentExecutor
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory
import prompts


def main():
    db.init_db()

    llm =  ChatGoogleGenerativeAI(model=config.LLM_MODEL, api_key=config.GOOGLE_API_KEY) 

    tools = get_tools_list()

    try:
        retriever, retriever_tool = build_retriever_from_urls(["https://docs.smith.langchain.com/"])
        tools.append(retriever_tool)
    except Exception as e:
        print("Warning: retriever build failed:", e)

    # custom vector memory
    # memory = VectorMemory()
    # memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    
    agent=create_react_agent(
        tools=tools,
        llm=llm,
        prompt=prompts.prompt

    )

    memory = ConversationBufferWindowMemory(
            k=10,  # Remember last 10 exchanges
            memory_key="chat_history",
            return_messages=True
        )
    
    past_logs = db.fetch_logs()

    for log in past_logs:
        user_input = log["user_input"]
        agent_response = log["agent_response"]
        # Save each pair into memory context
        memory.save_context({"input": user_input}, {"output": agent_response})


    
    agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            # max_iterations=3
        )

#

    print("Smart Research Assistant — type 'exit' to quit")
    while True:
        query = input("\nUser> ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            break
        res = agent_executor.invoke({"input": query })
        answer = res.get("output") or res.get("result") or str(res)
        print("\nAssistant>", answer)

        db.log_conversation(query, answer)

if __name__ == "__main__":
    main()
