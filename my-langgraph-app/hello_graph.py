from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from typing import TypedDict, List

load_dotenv()

class State(TypedDict):
    messages: List

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
)

def call_model(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

graph = StateGraph(State)
graph.add_node("agent", call_model)
graph.set_entry_point("agent")
graph.add_edge("agent", END)
app = graph.compile()

result = app.invoke({
    "messages": [HumanMessage(content="Hello! What is LangGraph in 3 sentences?")]
})
print(result["messages"][-1].content)
