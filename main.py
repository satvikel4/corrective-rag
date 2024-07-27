import os
from dotenv import load_dotenv
from config import setup_environment
from retrieval import create_retriever
from generation import create_rag_chain, create_retrieval_grader
from graph import create_workflow
from evaluation import predict_custom_agent_answer, create_dataset, run_evaluation

load_dotenv()
setup_environment()

def main():
    retriever = create_retriever()
    rag_chain = create_rag_chain()
    retrieval_grader = create_retrieval_grader()
    
    workflow = create_workflow(retriever, rag_chain, retrieval_grader)
    
    example = {"input": "What are the types of agent memory?"}
    response = predict_custom_agent_answer(example, workflow)
    print(response)
    
    create_dataset()
    run_evaluation()

if __name__ == "__main__":
    main()