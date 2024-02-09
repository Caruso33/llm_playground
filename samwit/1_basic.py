from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama

llm = Ollama(
    model="llama2-uncensored", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

llm.invoke("Tell me 5 facts about Roman history:")
