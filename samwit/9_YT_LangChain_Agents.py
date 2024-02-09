from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_community.llms import Ollama
from langchain_openai import OpenAI

load_dotenv()


def main():
    llm = Ollama(temperature=0, model="mistral:instruct")
    # llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")

    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    # print("Tool info: ", tools[1].name, tools[1].description)

    prompt = hub.pull("hwchase17/react")
    print(f"prompt {prompt}\n")

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    input = "What is langchain?"
    input += "Only use a tool if needed, otherwise respond with Final Answer"

    agent_executor.invoke(
        {"input": input},
    )
    return

    agent_executor.invoke(
        "Who is the United States President? What is his current age raised divided by 2?"
    )

    agent_executor.run(
        "What is the average age in the United States? How much less is that then the age of the current US President?"
    )

    # ## New Agent

    tools = load_tools(["serpapi", "llm-math", "wikipedia", "terminal"], llm=llm)

    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True
    )

    agent_executor.agent_executor.llm_chain.prompt.template

    agent_executor.run("Who is the head of DeepMind")

    agent_executor.run("What is DeepMind")

    agent_executor.run(
        "Take the year DeepMind was founded and add 50 years. Will this year have AGI?"
    )

    agent_executor.run("Where is DeepMind's office?")

    agent_executor.run(
        "If I square the number for the street address of DeepMind what answer do I get?"
    )

    agent_executor.run("What files are in my current directory?")

    agent_executor.run("Does my current directory have a file about California?")


if __name__ == "__main__":
    main()
