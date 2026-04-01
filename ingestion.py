import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
#from pinecone import Pinecone

load_dotenv()
#pc= Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = os.environ.get("INDEX_NAME")

if __name__ == '__main__':
    print("Loading data...")
    #print(os.getenv("PINECONE_API_KEY"))
    loader = TextLoader("mediumblog1.txt", encoding='utf-8')
    documents = loader.load()

    print("Splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks")
    
    
    print("Embedding...")
    embeddings = OpenAIEmbeddings()
    #vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    
    print("Ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=index_name)
    print("Done!")

