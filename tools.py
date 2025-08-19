import config
from langchain_community.tools import tool
from langchain_community.tools import Tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_community.vectorstores import FAISS
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
import datetime
from langchain.schema import HumanMessage
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from langchain.embeddings.base import Embeddings
from langchain.tools.retriever import create_retriever_tool
import gc

def weather(city:str)->str:

    try:
        res=requests.get(f"https://wttr.in/{city}?format=3", timeout=10)
        return res.text
    except Exception as e:
        return f"Error fetching weather: {e}"
    

tool_weather=Tool(
    name="WeatherTool",
    func=weather,
    description="Use to get weather/climate for a place"
)

@tool
def tool_news(query:str)->str:
    "Use when asked about news or current affairs"
    url = "https://api.thenewsapi.com/v1/news/top"
    params = {
        "api_token": config.NEWS_API_KEY,
        "search": query,
        "language": "en",
        "published_on": datetime.date.today().strftime("%Y-%m-%d"),
        "headlines_per_category": 5
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"Error fetching news: {response.text}"
    data = response.json().get("data", [])
    if not data:
        return "No news found for that query today."
    return "\n\n".join([f"📰 {item['title']} ({item.get('source')})\n{item['url']}" for item in data])


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash") 

def casual_chat_fn(query: str) -> str:
    return llm([HumanMessage(content=query)]).content

tool_chatbot = Tool(
    name="CasualChat",
    func=casual_chat_fn,
    description="Answer casual questions or chit-chat with the user."
)

wiki_api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
tool_wiki= WikipediaQueryRun(api_wrapper= wiki_api_wrapper)
   

arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
tool_arxiv= ArxivQueryRun(api_wrapper=arxiv_wrapper)
   

duckduckgo_wrapper=DuckDuckGoSearchAPIWrapper()
tool_search=DuckDuckGoSearchRun(api_wrapper=duckduckgo_wrapper)


sp_oauth = SpotifyOAuth(
    client_id=config.SPOTIFY_CLIENT_ID,
    client_secret=config.SPOTIFY_CLIENT_SECRET,
    redirect_uri=config.SPOTIFY_URI,
    scope="user-library-read user-read-playback-state streaming playlist-modify-public"
)

sp = spotipy.Spotify(auth_manager=sp_oauth)

@tool
def tool_spotify(query: str) -> str:
    """
    Search for tracks on Spotify by a query string.
    """
    results = sp.search(q=query, type='track', limit=3)
    tracks = results.get('tracks', {}).get('items', [])
    if not tracks:
        return "No tracks found."
    response = "Top tracks:\n"
    for i, track in enumerate(tracks, 1):
        artists = ', '.join(artist['name'] for artist in track['artists'])
        response += f"{i}. {track['name']} by {artists}\n"
    return response

class sentence_transformer(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

def build_pdf_retriever_tool(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embedding_model = sentence_transformer(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedding_model)
    retriever = db.as_retriever()
    del chunks
    del embedding_model
    gc.collect()
    return create_retriever_tool(
        retriever,
        name="pdf_retriever",
        description="Search for answers in your uploaded PDF document."
    )

def get_tools_list():
    return [tool_weather,tool_news,tool_chatbot,tool_wiki,tool_arxiv,tool_search,tool_spotify]