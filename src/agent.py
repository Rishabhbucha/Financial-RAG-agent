import os
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .tools import finrag_tools

# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iterations: int

# --- Critique Schema ---
class Critique(BaseModel):
    score: float = Field(description="A score between 0.0 and 1.0 evaluating the quality of the answer.")
    feedback: str = Field(description="Specific feedback on how to improve the answer based on faithfulness, relevance, and completeness.")

# --- Nodes ---
def call_agent(state: AgentState):
    """
    The main answering agent that uses tools to fetch context and generate a draft answer.
    """
    messages = state['messages']
    
    # We use a ReAct style approach. Langchain's bind_tools handles the tool calling natively.
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0) # Use a capable model
    llm_with_tools = llm.bind_tools(finrag_tools)
    
    # Optional: We could inject a system message here if not already present
    if not any(isinstance(m, SystemMessage) for m in messages):
        sys_prompt = (
            "You are Fin, an advanced financial AI assistant. "
            "CRITICAL: You HAVE LIVE TOOLS attached. "
            "NEVER say you cannot access live data, browse the internet, or create charts. "
            "ALWAYS use `search_financial_news` for web searches, and `get_stock_quote` for current prices. "
            "If the user asks for a chart, graph, or visual trend, you MUST call the `plot_stock_history` tool (pass the ticker and number of days, e.g., 180 for 6 months). The tool will create the chart and pop it up for the user automatically."
        )
        sys_msg = SystemMessage(content=sys_prompt)
        messages = [sys_msg] + list(messages)

    response = llm_with_tools.invoke(messages)
    
    # Check if tool calls were made
    if response.tool_calls:
        # We need to execute tools. For simplicity, we'll return the AIMessage containing tool calls.
        # The LangGraph will loop back via a ToolNode (we'll define this below).
        return {"messages": [response], "iterations": state.get('iterations', 0)}
    else:
        # It's a final draft answer
        return {"messages": [response], "iterations": state.get('iterations', 0)}


def critique_answer(state: AgentState):
    """
    Evaluates the draft answer.
    """
    messages = state['messages']
    
    # Extract the user's original query and the agent's latest answer
    original_query = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
    draft_answer = messages[-1].content
    
    eval_sys_prompt = """You are an expert financial reviewer. 
    Evaluate the draft answer based on:
    1. Faithfulness (no hallucinations)
    2. Relevance to the original question
    3. Completeness
    
    Provide a score between 0.0 and 1.0, and feedback for improvement."""
    
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    structured_llm = llm.with_structured_output(Critique)
    
    evaluation = structured_llm.invoke([
        SystemMessage(content=eval_sys_prompt),
        HumanMessage(content=f"Question: {original_query}\n\nDraft Answer:\n{draft_answer}")
    ])
    
    print(f"\n[Critique] Score: {evaluation.score} - Feedback: {evaluation.feedback}")
    
    # We add the feedback as a HumanMessage simulating the "Critic" telling the agent to fix it.
    feedback_msg = HumanMessage(content=f"Your previous answer received a score of {evaluation.score}. Feedback: {evaluation.feedback}\nPlease rewrite your answer incorporating this feedback.")
    
    return {
        "messages": [feedback_msg], 
        "iterations": state.get('iterations', 0) + 1,
        "latest_score": evaluation.score
    }

# --- Tool Execution Node ---
from langgraph.prebuilt import ToolNode
tool_node = ToolNode(finrag_tools)

# --- Edges & Routing ---
def route_agent_output(state: AgentState):
    """
    Decides whether to execute tools or proceed to critique.
    """
    last_message = state['messages'][-1]
    
    if last_message.tool_calls:
        return "tools"
    else:
        return "critique"
        
def route_critique_output(state: AgentState):
    """
    Decides whether the answer is good enough or needs rewrite.
    """
    # We passed 'latest_score' implicitly in the state, but TypedDict doesn't have it.
    # We can infer it by checking the iterations or we can modify AgentState to include it.
    # Let's just use iterations for now to prevent infinite loops.
    
    iterations = state.get('iterations', 0)
    
    # We need to extract the score from the last critique message.
    # Alternatively, just stop after 3 iterations. 
    # Let's say if it reached here, we just check if it's the 3rd iteration.
    if iterations >= 3:
        return END
        
    # How do we know if it passed? The last message is the "feedback_msg" we generated in critique_answer.
    last_msg = state['messages'][-1].content
    if "score of 1.0" in last_msg or "score of 0.9" in last_msg or "score of 0.85" in last_msg: 
        # Hacky check for simplicity. If score >= 0.85, we can end.
        # But wait, if it passed, we shouldn't have added the feedback message instructing a rewrite.
        return END
    
    return "agent"

# Let's slightly rewrite the routing to be cleaner.
# In `critique_answer`, we can return a specific flag in the state. Let's add it to TypedDict.

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iterations: int
    is_valid: bool

def critique_answer_v2(state: AgentState):
    messages = state['messages']
    original_query = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
    draft_answer = messages[-1].content
    
    # Using explicit JSON instructions instead of tool-calling to prevent formatting errors
    eval_sys_prompt = """You are an expert financial reviewer. Evaluate the draft answer for faithfulness, relevance, and completeness.
    
    You MUST respond with ONLY a valid JSON object in this exact format:
    {
        "score": 0.85,
        "feedback": "Your detailed feedback here."
    }
    Do NOT include any markdown formatting, backticks, or other text."""
    
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    
    try:
        response = llm.invoke([
            SystemMessage(content=eval_sys_prompt),
            HumanMessage(content=f"Question: {original_query}\n\nDraft Answer:\n{draft_answer}")
        ])
        
        import json
        content = response.content.strip()
        # Clean up any markdown blocks if the model ignored instructions
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        evaluation = json.loads(content)
        score = float(evaluation.get("score", 0.0))
        feedback = evaluation.get("feedback", "No feedback provided.")
        
    except Exception as e:
        print(f"\n[Critique Parsing Error] {e}")
        score = 0.5
        feedback = "The reviewer failed to format the response correctly. Please try answering again more accurately."
        
    print(f"\n[Critique] Score: {score} - Feedback: {feedback}")
    
    is_valid = score >= 0.85
    iterations = state.get('iterations', 0) + 1
    
    if not is_valid:
        feedback_msg = HumanMessage(content=f"CRITIQUE (Score: {score}): {feedback}\nPlease rewrite your answer.")
        return {"messages": [feedback_msg], "iterations": iterations, "is_valid": is_valid}
    else:
        # If valid, we don't necessarily need to add a message, just update state
        return {"iterations": iterations, "is_valid": is_valid}

def route_critique_v2(state: AgentState):
    if state.get("is_valid", False):
        return END
    if state.get("iterations", 0) >= 3:
        print("\n[Critique] Max iterations reached. Returning best effort.")
        return END
    return "agent"


# --- Build Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_agent)
workflow.add_node("tools", tool_node)
workflow.add_node("critique", critique_answer_v2)

# Set entry point
workflow.set_entry_point("agent")

# Add conditional edges from agent
workflow.add_conditional_edges(
    "agent",
    route_agent_output,
    {
        "tools": "tools",
        "critique": "critique"
    }
)

# After tools, always return to agent
workflow.add_edge("tools", "agent")

# After critique, conditionally loop back or end
workflow.add_conditional_edges(
    "critique",
    route_critique_v2,
    {
        "agent": "agent",
        END: END
    }
)

# Compile the graph
finrag_agent = workflow.compile()
