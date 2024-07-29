import os

def setup_environment():
    """
    Set up the environment variables for LangChain tracing and project configuration.
    
    This function sets the following environment variables:
    - LANGCHAIN_TRACING_V2: Enables version 2 of LangChain tracing
    - LANGCHAIN_ENDPOINT: Specifies the endpoint for LangChain API
    - LANGCHAIN_PROJECT: Sets the project name for LangChain
    """
    
    # Enable LangChain tracing version 2
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    
    # Set the LangChain API endpoint
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
    # Set the LangChain project name
    os.environ["LANGCHAIN_PROJECT"] = "corrective-rag"