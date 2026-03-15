from openai_agents import Agent, Runner

HYPOTHESIS_INSTRUCTIONS = "Given the information gathered, form a hypothesis. Return 'Complete' when done."
ONLINE_SEARCH_INSTRUCTIONS = "Given the information gathered, research online for relative topics, including the latest developments and benchmarks. \
    Return 'Complete' when done."
EXPERIMENT_INSTRUCTIONS = "Given the information gathered, write code in the existing database to run experiments. Return 'Complete' when done."
ANALYSIS_INSTRUCTIONS = "Given the experiments, analyze the results. If it matches the hypothesis, then write a paper, other wise rethink \
    on the experiments or the research."
PAPER_WRITER_INSTRUCTION = "Given the hypothesis, experiments, and analysis, write a research paper"
COORDINATOR_INSTRUCTIONS = """You are a researcher for a new topic. Find a topic to research by researching online, then formulate a research \
    hypothesis, write code to set up experiemnts, verify results, and write a paper.
"""

# Agents
hypothesis_agent = Agent(
    name="HypothesisAgent",
    instruction=HYPOTHESIS_AGENT_INSTRUCTIONS
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

# Distribute tools
hypothesis_agent.tools = [back_to_coordinator]
online_search_agent.tools = [research_online, back_to_coordinator]
experiment_agent.tools = [setup_codebase, back_to_coordinator]
analysis_agent.tools = [back_to_coordinator]
paper_writer_agent.tools = [back_to_coordinator]
coordinator_agent.tools = [transfer_to_hypothesis, transfer_to_online_search, transfer_to_experiment, \
    transfer_to_analysis, transfer_to_paper_writer]

if __name__ == "__main__":
    design_query = "Prove if there is an infinite number of prime numbers of the form x^2 + 1"
    
    result = Runner.run(coordinator_agent, design_query)
    
    print(f"\nFinal Report: {result.final_message}")

