import asyncio
from openai_agents import Agent, Runner, input_guardrail, GuardrailFunctionOutput, InputGuardrailTripwireTriggered
from pydantic import BaseModel

HYPOTHESIS_INSTRUCTIONS = "Given the information gathered, form a hypothesis. Return 'Complete' when done."
ONLINE_SEARCH_INSTRUCTIONS = "Given the information gathered, research online for relative topics, including the latest developments and benchmarks. \\\n    Return 'Complete' when done."
EXPERIMENT_INSTRUCTIONS = "Given the information gathered, write code in the existing database to run experiments. Return 'Complete' when done."
ANALYSIS_INSTRUCTIONS = "Given the experiments, analyze the results. Report the analysis back to the coordinator."
PAPER_WRITER_INSTRUCTION = "Given the hypothesis, experiments, and analysis, write a research paper"
POLITICAL_SAFETY_INSTRUCTIONS = "Determine if the research has gone towards a political direction or is becoming political sensitive."

COORDINATOR_INSTRUCTIONS = """You are a researcher for a new topic. Find a topic to research by researching online, then formulate a research \
    hypothesis, write code to set up experiemnts, verify results, and write a paper.
"""

# Agents
hypothesis_agent = Agent(
    name="HypothesisAgent",
    instruction=HYPOTHESIS_INSTRUCTIONS
)

online_search_agent = Agent(
    name="OnlineSearchAgent",
    instruction=ONLINE_SEARCH_INSTRUCTIONS
)

experiment_agent = Agent(
    name="ExperimentAgent",
    instruction=EXPERIMENT_INSTRUCTIONS
)

analysis_agent = Agent(
    name="AnalysisAgent",
    instruction=ANALYSIS_INSTRUCTIONS
)

paper_writer_agent = Agent(
    name="PaperWriterAgent",
    instruction=PAPER_WRITER_INSTRUCTION
)

coordinator_agent = Agent(
    name="CoordinatorAgent",
    instruction=COORDINATOR_INSTRUCTIONS
)

# Tools
def transfer_to_hypothesis():
    return hypothesis_agent

def transfer_to_online_search():
    return online_search_agent

def transfer_to_experiment():
    return experiment_agent

def transfer_to_analysis():
    return analysis_agent

def transfer_to_paper_writer():
    return paper_writer_agent

def back_to_coordinator():
    return coordinator_agent

def setup_codebase():
    # Some code to set up a codebase
    return None

def research_online():
    # Some method to search internet
    return None

# Guardrails
class PoliticsAnalysis(BaseModel):
    is_political: bool
    reasoning: str

political_safety_agent = Agent(
    name="PoliticalSafetyAgent",
    instruction=POLITICAL_SAFETY_INSTRUCTIONS,
    output_type=PoliticsAnalysis
)

@input_guardrail
async def block_politics(ctx, agent, user_input):
    result = Runner.run(political_safety_agent, user_input, context=ctx.content)
    check = await result.final_output

    return GuardrailFunctionOutput(
        tripwire_triggered=check.is_political,
        output_info=check.reasoning
    )

coordinator_agent.input_guardrails = [block_politics]


# Distribute tools
hypothesis_agent.tools = [back_to_coordinator]
online_search_agent.tools = [research_online, back_to_coordinator]
experiment_agent.tools = [setup_codebase, back_to_coordinator]
analysis_agent.tools = [back_to_coordinator]
paper_writer_agent.tools = [back_to_coordinator]
coordinator_agent.tools = [transfer_to_hypothesis, transfer_to_online_search, transfer_to_experiment, \
    transfer_to_analysis, transfer_to_paper_writer]

async def main():
    try:
        research_query = "Compare the time complexity of bubble sort and quicksort."
        result = await Runner.run(coordinator_agent, research_query)
        print(f"\nFinal Report: {result.final_message}")
    except InputGuardrailTripwireTriggered as e:
        print(f"Tripwire triggered: {e}")


if __name__ == "__main__":
    asyncio.run(main())