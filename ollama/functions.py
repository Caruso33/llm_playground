# https://python.langchain.com/docs/integrations/chat/ollama_functions

from langchain_experimental.llms.ollama_functions import OllamaFunctions

model = OllamaFunctions(model="llama2-uncensored")


model = model.bind(
    functions=[
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, " "e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        }
    ],
    function_call={"name": "get_current_weather"},
)

response = model.invoke("what is the weather in Boston?")

print(response)

###

from langchain.chains import create_extraction_chain
from langchain.schema import HumanMessage

# Schema
schema = {
    "properties": {
        "name": {"type": "string"},
        "height": {"type": "integer"},
        "hair_color": {"type": "string"},
    },
    "required": ["name", "height"],
}
messages = [
    HumanMessage(
        content="""Alex is 5 feet tall. Claudia is 1 feet taller than Alex and jumps higher than him. Claudia is a brunette and Alex is blonde."""
    )
]

input = messages

# Run chain
llm = OllamaFunctions(model="llama2-uncensored", temperature=0)

chain = create_extraction_chain(schema, llm)

response = chain.invoke(input)

print(response)
