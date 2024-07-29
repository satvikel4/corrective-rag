from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# Create a RAG (Retrieval-Augmented Generation) chain
def create_rag_chain():
    # Define the prompt template for the question-answering task
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.
        Use the following documents to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise:
        Question: {question}
        Documents: {documents}
        Answer:
        """,
        input_variables=["question", "documents"],
    )

    # Initialize the ChatOllama model
    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
    )

    # Chain together the prompt, language model, and string output parser
    return prompt | llm | StrOutputParser()

# Create a retrieval grader to assess document relevance
def create_retrieval_grader():
    # Initialize the ChatOllama model for grading
    llm = ChatOllama(model="llama3.1",
                     format="json",
                     temperature=0)

    # Define the prompt template for grading document relevance
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["question", "document"],
    )

    # Chain together the prompt, language model, and JSON output parser
    return prompt | llm | JsonOutputParser()