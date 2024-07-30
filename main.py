import os
from dotenv import load_dotenv
from config import setup_environment
from retrieval import create_retriever
from generation import create_rag_chain, create_retrieval_grader
from graph import create_workflow
from evaluation import predict_custom_agent_answer, create_dataset, run_evaluation

# Load environment variables from .env file
load_dotenv()
# Set up the environment for LangChain
setup_environment()

def main():
    # Create the retriever for fetching relevant documents
    retriever = create_retriever()
    # Create the RAG chain for generating answers
    rag_chain = create_rag_chain()
    # Create the retrieval grader for assessing document relevance
    retrieval_grader = create_retrieval_grader()
    
    # Create the workflow combining retriever, RAG chain, and grader
    workflow = create_workflow(retriever, rag_chain, retrieval_grader)
    
    # Define an example question
    example = {"input": "What are the types of agent memory?"}
    # Predict the answer using the custom agent
    response = predict_custom_agent_answer(example, workflow)
    # Print the response
    print(response)
    
    # Create the dataset for evaluation
    create_dataset()
    # Run the evaluation process
    run_evaluation()

# Execute the main function if this script is run directly
if __name__ == "__main__":
    main()