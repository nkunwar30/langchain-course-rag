import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    #print(os.getenv("PINECONE_API_KEY"))
    loader = TextLoader("mediumblog1.txt")
    documents = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks")
    
    
    print("Embedding...")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    vectorstore = PineconeVectorStore(index_name="mediumblog1", embedding=embeddings)
    
    print("Ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=''
    print("Done!")

