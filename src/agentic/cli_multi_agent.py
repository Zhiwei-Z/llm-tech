import asyncio
import os
import subprocess
import sys
from typing import Union
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import Agent, Runner, function_tool, ModelSettings

load_dotenv()



# --- Tools ---

@function_tool
def execute_cli(command: str) -> str:
    """Executes a specific CLI command in the local shell."""
    print(f"\n[APPROVAL REQUIRED]")
    print(f"Proposed Command: {command}")
    
    choice = input("Execute this command? (y/n): ").strip().lower()
    if choice != 'y':
        return "ERROR: User rejected this command."

    print("[*] Running...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    except Exception as e:
        return f"Error executing command: {str(e)}"

@function_tool
def propose_plan(plan: str) -> Union[str, Agent]: # Note the Union type
    print("\n" + "="*30)
    print("PROPOSED ARCHITECTURAL PLAN:")
    print(plan)
    print("="*30)
    
    choice = input("\nDo you approve this plan? (y/n): ").strip().lower()
    
    if choice == 'y':
        print("[*] Plan approved. Transferring to CLI-Agent...")
        return cli_agent  # Return the agent object directly to force the handoff
    else:
        feedback = input("Provide feedback for the Architect: ")
        return f"REJECTED: User did not approve. Feedback: {feedback}"

# --- Handover Functions ---

@function_tool
def transfer_to_cli_agent():
    """Transfer to the CLI agent to execute the approved plan."""
    return cli_agent

@function_tool
def transfer_to_architect():
    """Transfer back to the architect for further planning or summary."""
    return architect_agent

# --- Agent Definitions ---

model_settings = ModelSettings(parallel_tool_calls=False)

cli_agent = Agent(
    name="CLI-Agent",
    instructions="""You are an execution specialist.
    1. Execute the commands provided in the plan one by one.
    2. If a command fails, report it and transfer back to the architect.
    3. Once all commands in the current phase are done, transfer back to the architect.""",
    tools=[execute_cli, transfer_to_architect],
    model="gpt-4o",
    model_settings=model_settings
)

architect_agent = Agent(
    name="Architect",
    instructions="""You are the lead system designer and planner.
    When you need to acquire more context, propose using ls commands and git commands.
    1. Analyze the user request and the current workspace context.
    2. Create a step-by-step plan.
    3. PRESENT the plan to the user and ask for explicit approval
    4. Transfer to cli agent when the plan is approved.
    4. After the CLI-Agent finishes, verify the results and provide a final summary.""",
    tools=[propose_plan, transfer_to_cli_agent],
    model="gpt-4o",
    model_settings=model_settings
)

# --- Execution ---

def main():
    if len(sys.argv) < 2:
        print("Usage: python agent.py 'your instruction'")
        return

    user_input = " ".join(sys.argv[1:])
    
    # We start with the Architect Agent
    result = Runner.run_sync(architect_agent, user_input)
    
    print("\n--- Final System Response ---")
    print(result.final_output)

if __name__ == "__main__":
    main()