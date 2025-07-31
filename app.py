from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import AsyncGenerator
import asyncio
import json
import os
from dotenv import load_dotenv

# LangChain imports
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from tavily import TavilyClient

# Load environment variables
load_dotenv()

app = FastAPI()

# Agent State Definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    search_results: str
    query_count: int

# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def tavily_search(query: str) -> str:
    """Search the web using Tavily for current, relevant information."""
    try:
        response = tavily_client.search(query=query, max_results=3)
        if response and 'results' in response:
            formatted_results = []
            for i, result in enumerate(response['results'], 1):
                formatted_result = f"""
                Result {i}:
                Title: {result.get('title', 'N/A')}
                Content: {result.get('content', 'N/A')[:300]}...
                URL: {result.get('url', 'N/A')}
                """
                formatted_results.append(formatted_result)
            return "\n".join(formatted_results)
        else:
            return "No search results found."
    except Exception as e:
        return f"Search error: {str(e)}"

# Initialize models
basic_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tavily_tools = [tavily_search]
tavily_model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tavily_tools)

# Basic Agent (Fixed system prompt)
def basic_agent_node(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="""
    You are a POWERFUL AI research assistant.
    
    Your capabilities include:
    - Access to current events and breaking news
    - Real-time data and statistics
    - Recent developments in any field
    - Up-to-date information on any topic
    
    Provide comprehensive, well-sourced answers..

    
    IMPORTANT: Format your responses using markdown:
    - Use ## for main section headings
    - Use ### for subsection headings
    - Use **bold** for key terms and important points
    - Use numbered lists (1., 2., 3.) for sequential steps or ranked items
    - Use bullet points (-) for feature lists or examples
    - Use `code` for technical terms
    - Add blank lines between sections for better spacing
    - Structure information hierarchically with clear sections
    """)
    
    messages = [system_prompt] + list(state["messages"])
    response = basic_model.invoke(messages)
    
    return {
        "messages": [response],
        "search_results": state.get("search_results", ""),
        "query_count": state.get("query_count", 0) + 1
    }

# Tavily Agent
def tavily_agent_node(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="""
    You are a POWERFUL AI research assistant with access to real-time web search via Tavily.
    
    Your capabilities include:
    - Access to current events and breaking news
    - Real-time data and statistics
    - Recent developments in any field
    - Up-to-date information on any topic
    
    When to use Tavily search:
    - Questions about current events, news, or recent developments
    - Requests for latest data, statistics, or market information
    - Any query that might benefit from real-time information
    - When your training data might be outdated
    
    Always search first when the query involves:
    - "Current", "latest", "recent", "today", "now"
    - Specific dates after your training cutoff
    - Stock prices, weather, news, sports scores
    - Company updates, product releases, events
    
    IMPORTANT: Format your responses using markdown for maximum readability:
    - Use ## for main section headings
    - Use ### for subsection headings
    - Use **bold** for key terms and important points
    - Use numbered lists (1., 2., 3.) for sequential steps or ranked items
    - Use bullet points (-) for feature lists or examples
    - Use `code` for technical terms
    - Include links when referencing sources
    - Add blank lines between sections for better spacing
    - Structure information hierarchically with clear sections
    
    Provide comprehensive, well-sourced answers using the search results.
    """)
    
    messages = [system_prompt] + list(state["messages"])
    response = tavily_model.invoke(messages)
    
    return {
        "messages": [response],
        "search_results": state.get("search_results", ""),
        "query_count": state.get("query_count", 0) + 1
    }

def should_continue_tavily(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    else:
        return "end"

# Build graphs
basic_graph = StateGraph(AgentState)
basic_graph.add_node("agent", basic_agent_node)
basic_graph.set_entry_point("agent")
basic_graph.add_edge("agent", END)
basic_app = basic_graph.compile()

tavily_graph = StateGraph(AgentState)
tavily_graph.add_node("agent", tavily_agent_node)
tavily_graph.add_node("tools", ToolNode(tavily_tools))
tavily_graph.set_entry_point("agent")
tavily_graph.add_conditional_edges(
    "agent",
    should_continue_tavily,
    {
        "continue": "tools",
        "end": END
    }
)
tavily_graph.add_edge("tools", "agent")
tavily_app = tavily_graph.compile()

async def stream_agent_response(agent_type: str, query: str) -> AsyncGenerator[str, None]:
    """Stream agent responses with thinking indicators"""
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "search_results": "",
        "query_count": 0
    }
    
    # Send thinking indicator
    yield f"data: {json.dumps({'type': 'thinking', 'agent': agent_type})}\n\n"
    await asyncio.sleep(0.5)
    
    try:
        if agent_type == "basic":
            result = basic_app.invoke(initial_state)
        else:
            result = tavily_app.invoke(initial_state)
        
        response_content = result["messages"][-1].content
        
        # Send typing indicator
        yield f"data: {json.dumps({'type': 'typing', 'agent': agent_type})}\n\n"
        await asyncio.sleep(0.3)
        
        # Stream words one by one
        words = response_content.split()
        for i, word in enumerate(words):
            yield f"data: {json.dumps({'type': 'word', 'agent': agent_type, 'word': word, 'is_last': i == len(words)-1})}\n\n"
            await asyncio.sleep(0.05)  # Adjust speed as needed
        
        # Send completion
        yield f"data: {json.dumps({'type': 'complete', 'agent': agent_type})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'agent': agent_type, 'error': str(e)})}\n\n"

@app.post("/chat/basic")
async def chat_basic(request: Request):
    data = await request.json()
    query = data.get("message", "")
    
    return StreamingResponse(
        stream_agent_response("basic", query),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/chat/tavily")
async def chat_tavily(request: Request):
    data = await request.json()
    query = data.get("message", "")
    
    return StreamingResponse(
        stream_agent_response("tavily", query),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.get("/")
async def get_chat_interface():
    with open("chat_interface.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 