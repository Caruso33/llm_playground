import argparse

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import Chroma


def main():
    parser = argparse.ArgumentParser(description="Filter out URL argument.")
    parser.add_argument(
        "--url",
        type=str,
        default="http://example.com",
        required=True,
        help="The URL to filter out.",
    )

    args = parser.parse_args()
    url = args.url
    print(f"using URL: {url}")

    loader = WebBaseLoader(url)
    data = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)
    print(f"Split into {len(all_splits)} chunks")

    model = "llama2-uncensored"
    embeddings = OllamaEmbeddings(model=model)
    # | GPT4AllEmbeddings

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

    # Retrieve
    # question = "What are the latest headlines on {url}?"
    # docs = vectorstore.similarity_search(question)

    print(f"Loaded {len(data)} documents")
    # print(f"Retrieved {len(docs)} documents")

    # RAG prompt
    from langchain import hub

    QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")
    # print(f"Prompt {QA_CHAIN_PROMPT}\n")
    # Prompt input_variables=['context', 'question'] messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> \nQuestion: {question} \nContext: {context} \nAnswer: [/INST]"))]

    # LLM
    llm = Ollama(
        model=model,
        temperature=0,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    print(f"Loaded LLM model {llm.model}")

    # QA chain
    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    # Ask a question
    question = f"What are the latest headlines on {url}? Max 100 words per headline and reference the url."
    result = qa_chain.invoke({"query": question})

    print(result)


if __name__ == "__main__":
    main()
