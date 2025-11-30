from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import os

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = (
        "Elon Musk is a renowned entrepreneur and inventor best known as the CEO of SpaceX and Tesla, Inc. "
        "Born in South Africa in 1971, he has played a pivotal role in advancing electric vehicles, "
        "renewable energy, and private space exploration. Musk also co-founded companies such as PayPal and Neuralink, "
        "and has been a strong advocate for ambitious projects like the Hyperloop and the colonization of Mars. "
        "His innovative vision and willingness to take risks have made him one of the most influential figures in technology today."
    )

    summary_template = """
    Given the following information about a person, extract a short summary:
    {information}
    """

    summary_prompt_template = ChatPromptTemplate.from_template(summary_template)
    # llm = ChatOpenAI(model="gpt-5", temperature=1)
    llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain = summary_prompt_template | llm  # runnable chain
    summary = chain.invoke({"information": information})
    print(summary.content)


if __name__ == "__main__":
    main()
