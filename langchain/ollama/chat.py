# https://python.langchain.com/docs/integrations/chat/ollama

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
llm = ChatOllama(model="llama2-uncensored")
prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")

# using LangChain Expressive Language chain syntax
# learn more about the LCEL on
# https://python.langchain.com/docs/expression_language/why
chain = prompt | llm | StrOutputParser()

# for brevity, response is printed in terminal
# You can use LangServe to deploy your application for
# production
print(chain.invoke({"topic": "Space travel"}))

####

llm = ChatOllama(model="llama2-uncensored", format="json", temperature=0)

from langchain.schema import HumanMessage

messages = [
    HumanMessage(
        content="What color is the sky at different times of the day? Respond using JSON"
    )
]

chat_model_response = llm.invoke(messages)
print(chat_model_response)


###

import json

from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

json_schema = {
    "title": "Person",
    "description": "Identifying information about a person.",
    "type": "object",
    "properties": {
        "name": {"title": "Name", "description": "The person's name", "type": "string"},
        "age": {"title": "Age", "description": "The person's age", "type": "integer"},
        "fav_food": {
            "title": "Fav Food",
            "description": "The person's favorite food",
            "type": "string",
        },
    },
    "required": ["name", "age"],
}

llm = ChatOllama(model="llama2-uncensored")

messages = [
    HumanMessage(
        content="Please tell me about a person using the following JSON schema:"
    ),
    HumanMessage(content="{dumps}"),
    HumanMessage(
        content="Now, considering the schema, tell me about a person named John who is 35 years old and loves pizza."
    ),
]

prompt = ChatPromptTemplate.from_messages(messages)
dumps = json.dumps(json_schema, indent=2)

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"dumps": dumps}))
