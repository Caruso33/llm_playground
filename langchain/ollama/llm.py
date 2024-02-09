# https://python.langchain.com/docs/integrations/llms/ollama

from langchain_community.llms import Ollama

llm = Ollama(model="llama2-uncensored")

# Tell me a very dirty quick joke
# How's Elon Musk?
instruction = """
What is the date of the latest source you have?
"""

response = llm.invoke(instruction)

print(f'{response}\n')