from supabase import create_client
import config

SUPABASE_URL = config.SUPABASE_URL
SUPABASE_KEY = config.SUPABASE_ANON_KEY

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def init_db():
    pass

def log_conversation(user_input: str, agent_response: str):
    data = { "user_input": user_input, "agent_response": agent_response }
    try:
        response = supabase.table("conversations").insert(data).execute()
        if response.data:
            print("Log inserted successfully", response.data)
        else:
            print("Insert executed but no data returned")
    except Exception as e:
        print("Failed to insert log:", e)

def fetch_logs():
    try:
        response = supabase.table("conversations").select("*").order("id", desc=True).limit(10).execute()
        if response.data:
            for row in reversed(response.data):
                print(f"{row['id']}: {row['user_input']} -> {row['agent_response']}")
            return reversed(response.data)
        else:
            print("No logs found")
            return []
    except Exception as e:
        print("Failed to fetch logs:", e)
        return []

