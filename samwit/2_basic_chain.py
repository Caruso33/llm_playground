from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama

llm = Ollama(
    model="llama2-uncensored",
    #  callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
    temperature=0.9,
)

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Give me 5 interesting facts about {topic}?",
)

from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt, verbose=False)

# Run the chain only specifying the input variable.
print(chain.invoke("the moon"))
