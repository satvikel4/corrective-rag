from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings

def create_retriever():
    # Define URLs to fetch documents from
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    # Load documents from each URL
    docs = [WebBaseLoader(url).load() for url in urls]
    # Flatten the list of documents
    docs_list = [item for sublist in docs for item in sublist]

    # Create a text splitter to chunk the documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    # Split the documents into smaller chunks
    doc_splits = text_splitter.split_documents(docs_list)

    # Create a vector store from the document splits
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings(),
    )
    # Return the vector store as a retriever, configured to fetch 4 most relevant documents
    return vectorstore.as_retriever(k=4)