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
        "answer": "To find the capital of France, I will use the Wikipedia Search tool.\n\nWikipedia Search: Paris\n\nParis is the capital of France."
    },
    {
        "query": "What is the sum of 2 and 3?",
        "answer": "The sum of 2 and 3 is 5. I don't need to use the Wikipedia Search tool for this simple arithmetic calculation."
    },
    {
        "query": "Who painted the Mona Lisa?",
        "answer": "To find out who painted the Mona Lisa, I will use the Wikipedia Search tool.\n\nWikipedia Search: Mona Lisa\n\nThe Mona Lisa was painted by the Italian Renaissance artist Leonardo da Vinci."
    },
    {
        "query": "What is the largest planet in our solar system?",
        "answer": "To find the largest planet in our solar system, I will use the Wikipedia Search tool.\n\nWikipedia Search: Largest planet\n\nJupiter is the largest planet in our solar system."
    },
    {
        "query": "What is the currency of Japan?",
        "answer": "To find the currency of Japan, I will use the Wikipedia Search tool.\n\nWikipedia Search: Japanese yen\n\nThe currency of Japan is the Japanese yen."
    },
    {
        "query": "Who wrote the play 'Romeo and Juliet'?",
        "answer": "To find out who wrote the play 'Romeo and Juliet', I will use the Wikipedia Search tool.\n\nWikipedia Search: Romeo and Juliet\n\nThe play 'Romeo and Juliet' was written by the English playwright William Shakespeare."
    },
    {
        "query": "What is the tallest mountain in the world?",
        "answer": "To find the tallest mountain in the world, I will use the Wikipedia Search tool.\n\nWikipedia Search: Mount Everest\n\nMount Everest is the tallest mountain in the world, with a peak elevation of 29,031.7 feet (8,848.86 meters) above sea level."
    },
    {
        "query": "Who is the current President of the United States?",
        "answer": "As of August 2023, the current President of the United States is Joe Biden. He took office on January 20, 2021."
    },
    {
        "query": "What is the chemical symbol for gold?",
        "answer": "The chemical symbol for gold is Au. I don't need to use the Wikipedia Search tool for this simple fact."
    },
    {
        "query": "What is the capital of Australia?",
        "answer": "To find the capital of Australia, I will use the Wikipedia Search tool.\n\nWikipedia Search: Canberra\n\nCanberra is the capital city of Australia."
    },
    {
        "query": "Who invented the telephone?",
        "answer": "To find out who invented the telephone, I will use the Wikipedia Search tool.\n\nWikipedia Search: Alexander Graham Bell\n\nAlexander Graham Bell is credited with inventing the first practical telephone."
    },
    {
        "query": "What is the largest continent by land area?",
        "answer": "To find the largest continent by land area, I will use the Wikipedia Search tool.\n\nWikipedia Search: Asia\n\nAsia is the largest continent by land area, covering approximately 44,579,000 square kilometers (17,212,000 square miles)."
    },
    {
        "query": "Who is the author of the Harry Potter book series?",
        "answer": "To find the author of the Harry Potter book series, I will use the Wikipedia Search tool.\n\nWikipedia Search: J. K. Rowling\n\nJ. K. Rowling is the author of the Harry Potter book series."
    },
    {
        "query": "What is the capital of Brazil?",
        "answer": "To find the capital of Brazil, I will use the Wikipedia Search tool.\n\nWikipedia Search: Brasília\n\nBrasília is the capital of Brazil."
    },
    {
        "query": "Who painted the famous artwork 'The Starry Night'?",
        "answer": "To find out who painted 'The Starry Night', I will use the Wikipedia Search tool.\n\nWikipedia Search: The Starry Night\n\n'The Starry Night' is a famous oil painting by the Dutch Post-Impressionist painter Vincent van Gogh."
    },
    {
        "query": "What is the largest ocean in the world?",
        "answer": "To find the largest ocean in the world, I will use the Wikipedia Search tool.\n\nWikipedia Search: Pacific Ocean\n\nThe Pacific Ocean is the largest and deepest ocean in the world, covering approximately 165,250,000 square kilometers (63,800,000 square miles)."
    },
    {
        "query": "Who wrote the novel 'To Kill a Mockingbird'?",
        "answer": "To find out who wrote the novel 'To Kill a Mockingbird', I will use the Wikipedia Search tool.\n\nWikipedia Search: Harper Lee\n\nThe novel 'To Kill a Mockingbird' was written by American author Harper Lee."
    },
    {
        "query": "What is the capital of Canada?",
        "answer": "To find the capital of Canada, I will use the Wikipedia Search tool.\n\nWikipedia Search: Ottawa\n\nOttawa is the capital city of Canada."
    },
    {
        "query": "Who is the founder of Microsoft?",
        "answer": "To find out who founded Microsoft, I will use the Wikipedia Search tool.\n\nWikipedia Search: Bill Gates\n\nBill Gates is the co-founder of Microsoft Corporation, along with Paul Allen."
    },
    {
        "query": "What is the chemical formula for water?",
        "answer": "The chemical formula for water is H2O. I don't need to use the Wikipedia Search tool for this simple fact."
    },
    {
        "query": "Who is the current Prime Minister of the United Kingdom?",
        "answer": "As of August 2023, the current Prime Minister of the United Kingdom is Rishi Sunak. He took office on October 25, 2022."
    },
    {
        "query": "What is the largest desert in the world?",
        "answer": "To find the largest desert in the world, I will use the Wikipedia Search tool.\n\nWikipedia Search: Antarctic Desert\n\nThe Antarctic Desert, which covers the continent of Antarctica, is the largest desert in the world, with an area of about 5.5 million square miles (14.2 million square kilometers)."
    },
    {
        "query": "Who painted the famous artwork 'The Persistence of Memory'?",
        "answer": "To find out who painted 'The Persistence of Memory', I will use the Wikipedia Search tool.\n\nWikipedia Search: Salvador Dalí\n\n'The Persistence of Memory' is a famous surrealist painting by Spanish artist Salvador Dalí."
    },
    {
        "query": "What is the capital of Russia?",
        "answer": "To find the capital of Russia, I will use the Wikipedia Search tool.\n\nWikipedia Search: Moscow\n\nMoscow is the capital city of Russia."
    },
    {
        "query": "Who wrote the play 'Hamlet'?",
        "answer": "To find out who wrote the play 'Hamlet', I will use the Wikipedia Search tool.\n\nWikipedia Search: William Shakespeare\n\nThe play 'Hamlet' was written by the English playwright William Shakespeare."
    },
    {
        "query": "What is the largest mammal in the world?",
        "answer": "To find the largest mammal in the world, I will use the Wikipedia Search tool.\n\nWikipedia Search: Blue whale\n\nThe blue whale is the largest mammal and the largest animal known to have ever existed, with an average length of around 24-30 meters (80-100 feet) and a weight of up to 200 tons."
    },
    {
        "query": "Who is the founder of Amazon?",
        "answer": "To find out who founded Amazon, I will use the Wikipedia Search tool.\n\nWikipedia Search: Jeff Bezos\n\nJeff Bezos is the founder and former CEO of Amazon, the multinational technology company."
    },
    {
        "query": "What is the chemical symbol for silver?",
        "answer": "The chemical symbol for silver is Ag. I don't need to use the Wikipedia Search tool for this simple fact."
    },
    {
        "query": "Who is the current Prime Minister of India?",
        "answer": "As of August 2023, the current Prime Minister of India is Narendra Modi. He assumed office on May 26, 2014."
    },
    {
        "query": "What is the tallest building in the world?",
        "answer": "To find the tallest building in the world, I will use the Wikipedia Search tool.\n\nWikipedia Search: Burj Khalifa\n\nThe Burj Khalifa in Dubai, United Arab Emirates, is currently the tallest building in the world, with a height of 828 meters (2,717 feet)."
    }
]

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory,
    handle_parsing_errors=True
)

query = "What is the capital of Germany?"
response = agent.run(query)
print(response)