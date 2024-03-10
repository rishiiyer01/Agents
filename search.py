#using claude3 (probably not opus) for search
from langchain.agents import initialize_agent, Tool, AgentExecutor

from langchain.memory import ConversationBufferMemory
import wikipedia

def search_wikipedia(query):
    try:
        result = wikipedia.summary(query, sentences=2)
        return result
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found for '{query}'. Please be more specific. Possible matches: {e.options}"
    except wikipedia.exceptions.PageError:
        return f"No results found for '{query}' on Wikipedia."

tools = [
    Tool(
        name="Wikipedia Search",
        func=search_wikipedia,
        description="Searches Wikipedia for relevant information."
    )
]

with open("/home/iyer.ris/anthropic_api_key.txt", "r") as file:
    api_key = file.read().strip()
from langchain_anthropic import AnthropicLLM

llm= AnthropicLLM(model='claude-2.1',anthropic_api_key=api_key)

examples = [
    {
        "query": "What is the capital of France?",
        "answer": "Thought: To find the capital of France, I need to search Wikipedia.\nAction: Wikipedia Search\nAction Input: Capital of France\nObservation: Paris is the capital and most populous city of France.\nThought: Based on the information from Wikipedia, the capital of France is Paris.\nFinal Answer: The capital of France is Paris."
    },
    {
        "query": "What is the largest planet in our solar system?",
        "answer": "Thought: To find the largest planet in our solar system, I will search Wikipedia.\nAction: Wikipedia Search\nAction Input: Largest planet in the solar system\nObservation: Jupiter is the largest planet in the solar system.\nThought: According to Wikipedia, Jupiter is the largest planet in our solar system.\nFinal Answer: Jupiter is the largest planet in our solar system."
    },
    {
        "query": "Who wrote the novel 'To Kill a Mockingbird'?",
        "answer": "Thought: To find the author of 'To Kill a Mockingbird', I will search Wikipedia.\nAction: Wikipedia Search\nAction Input: Author of 'To Kill a Mockingbird'\nObservation: 'To Kill a Mockingbird' is a novel by American author Harper Lee.\nThought: Based on the information from Wikipedia, the novel 'To Kill a Mockingbird' was written by Harper Lee.\nFinal Answer: The novel 'To Kill a Mockingbird' was written by Harper Lee."
    },
    {
        "query": "What is the chemical symbol for gold?",
        "answer": "Thought: The chemical symbol for gold is a common knowledge and doesn't require a Wikipedia search.\nFinal Answer: The chemical symbol for gold is Au."
    },
    {
        "query": "Who painted the famous artwork 'The Starry Night'?",
        "answer": "Thought: To find the painter of 'The Starry Night', I will search Wikipedia.\nAction: Wikipedia Search\nAction Input: Painter of 'The Starry Night'\nObservation: 'The Starry Night' is an oil-on-canvas painting by the Dutch Post-Impressionist painter Vincent van Gogh.\nThought: According to Wikipedia, the famous painting 'The Starry Night' was painted by Vincent van Gogh.\nFinal Answer: The famous artwork 'The Starry Night' was painted by Vincent van Gogh."
    }
]

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools,
    llm,
    agent="conversational-react-description",
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory,
    handle_parsing_errors=True
)

query = "What is the capital of Germany?"
response = agent.run(query)
print(response)