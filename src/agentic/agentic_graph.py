from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END

# This is the 'State' that every node will read and write to
class AgentState(TypedDict):
    content: str
    revision_count: int

# Define the nodes (functions that run)
def initial_generator(state: AgentState):
    print("---Generating Initial Answer---")
    return {"content": "The sky is blue because of Rayleigh scattering.", "revision_count": 1}

def add_flair(state: AgentState):
    print("---Adding Professional Flair---")
    new_content = state["content"] + " This is a fundamental principle of optics."
    return {"content": new_content, "revision_count": state["revision_count"] + 1}

# 1. Initialize the graph with our State schema
workflow = StateGraph(AgentState)

# 2. Add our nodes
workflow.add_node("generator", initial_generator)
workflow.add_node("refiner", add_flair)

# 3. Define the edges (the arrows)
workflow.add_edge(START, "generator") # Start here
workflow.add_edge("generator", "refiner") # Then go here
workflow.add_edge("refiner", END)        # Then finish

# 4. Compile the graph
app = workflow.compile()

final_state = app.invoke({"content": "", "revision_count": 0})

print("\nFinal Result:")
print(f"Content: {final_state['content']}")
print(f"Total Steps: {final_state['revision_count']}")
