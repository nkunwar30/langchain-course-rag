import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough




load_dotenv()
print("Initializaing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-5.2")

vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context}
    Question: {question}
    provide a detailed answer:"""
)

chain = (prompt_template | retriever | llm).with_config(run_name="rag_chain").with_config(verbose=True)

def format_docs(docs):
    """Format retrieved documents into a string for the prompt"""
    return "\n\n".join([doc.page_content for doc in docs])


# =====================================================================================
# Implementation 1:Without LCEL (Simple function based approach)
# =====================================================================================
def retrieval_chain_without_lcel(query=str):
    """
    Simple retrieval chain without LCEL,
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    #step 1: Retrieve relevant documents
    docs= retriever.invoke(query)

    # Step 2:Format document
    context=format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages=prompt_template.format_messages(context=context,question=query)

    #Step 4: Invoke LLM with formatted messages
    response = llm.invoke(messages)

    # step 5:Return only the content part of the LLM's response
    return response.content
# =====================================================================================
# Implementation 2: With LCEL (LangChain Expression Language)
# =====================================================================================
# This implementation leverages LangChain's Expression Language (LCEL)
# to compose the retrieval-augmented generation (RAG) pipeline as a single chain.
#
# Advantages:
# - More concise and declarative pipeline definition
# - Built-in streaming and async support
# - Easier to compose with other chains and components
# - Enhanced observability and debugging via cfg/WithConfig
# =====================================================================================
def create_retrieval_chain_with_lcel():
    ''' 
    Create a retrieval chain using LCEL (LangChain Expression Language), Returns a Chain
    that can be invoked with {"question":"....."}
    '''
    retrieval_chain =(RunnablePassthrough.assign(
        context=itemgetter("question")|retriever|format_docs)
        |prompt_template|llm|StrOutputParser())

    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")
    
    # Query
    query = "what is pinecone in machine learning ?"

    #===================================================================================
    # Option 0 : raw Invocation without RAG
    #===================================================================================
    print("\n" + "="*70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("="*70)
    result_raw=llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    #===================================================================================
    # Option 1 : raw Invocation without LCEL
    #===================================================================================
    print("\n" + "="*70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("="*70)
    result_without_lcel=retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    #===================================================================================
    # Option 2 : LCEL-Based RAG Invocation
    #===================================================================================
    # This illustrates using the retrieval chain composed via LCEL.
    # - More concise, declarative, and supports easy streaming/async.
    # - Good for advanced compositions and better debugging.
    # - Invoked with a dictionary input: {"question": ...}
    print("\n" + "="*70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("="*70)
    print("why LCEL is better")
    print("- More concise, declarative, and supports easy streaming/async")
    print("Built-in streaming: chain.stream() ")
    print("Built-in async: chain.ainvoke() ")
    print("Easy to compose with other chains ")
    print("- Better for production use ")
    print("="*70)
    
    chain_with_lcel=create_retrieval_chain_with_lcel()
    result_with_lcel=chain_with_lcel.invoke({"question":query})
    print("\nAnswer:")
    print(result_with_lcel)