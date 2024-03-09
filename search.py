#using claude3 (probably not opus) for search
from langchain.agents import initialize_agent, Tool
from langchain.llms import Anthropic
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
api_key=/home/iyer.ris/anthropic_api_key.txt
llm = Anthropic(api_key="api_key")

examples = [
    {
        "query": "What is the capital of France?",
        "answer": "To find the capital of France, I will use the Wikipedia Search tool.\n\nWikipedia Search: Paris\n\nParis is the capital of France."
    },
    {
        "query": "What is the sum of 2 and 3?",
        "answer": "The sum of 2 and 3 is 5. I don't need to use the Wikipedia Search tool for this simple arithmetic calculation."
    }
]

agent = initialize_agent(
    tools,
    llm,
    agent="conversational-react-description",
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    examples=examples
)

query = "What is the capital of Germany?"
response = agent.run(query)
print(response)