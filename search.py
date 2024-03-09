#using claude3 (probably not opus) for search

#using few shot, since I don't want search_wikipedia to be called every time that is annoying
from langchain.agents import initialize_agent, Tool
from langchain.llms import Anthropic
from langchain.agents import initialize_agent, Tool
from langchain.llms import Anthropic

def search_wikipedia(query):
    # Implement the Wikipedia search functionality here
    # Return the search result
    pass

tools = [
    Tool(
        name="Wikipedia Search",
        func=search_wikipedia,
        description="Searches Wikipedia for relevant information."
    )
]

llm = Anthropic(api_key="YOUR_API_KEY")

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