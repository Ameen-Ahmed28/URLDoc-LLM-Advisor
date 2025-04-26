import os
import time
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st

# Function to load and process documents from a URL
def get_docs_from_urls(urls):
    all_docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        split_docs = text_splitter.split_documents(docs)
        all_docs.extend(split_docs)
        st.sidebar.write(f"Documents loaded from URL: {url}")
    return all_docs

def get_docs(uploaded_file):
    start_time = time.time()
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(documents)
    st.sidebar.write('Documents Loaded')
    end_time = time.time()
    st.sidebar.write(f"Time taken to load documents: {end_time - start_time:.2f} seconds")
    os.remove("temp.pdf")  # Clean up the temporary file
    return final_documents

# Function to create vector store
def create_vector_store(docs):
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"trust_remote_code": True})
    vectorstore = FAISS.from_documents(docs, embeddings)
    st.sidebar.write('DB is ready 💾')
    end_time = time.time()
    st.sidebar.write(f"Time taken to create DB: {end_time - start_time:.2f} seconds")
    return vectorstore

# Function to interact with Groq AI
def chat_groq(messages):
    load_dotenv()
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    response_content = ''
    stream = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        max_tokens=1024,
        temperature=1.3,
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            response_content += chunk.choices[0].delta.content
    return response_content

# Function to summarize chat history
def summarize_chat_history(chat_history):
    chat_history_text = " ".join([f"{chat['role']}: {chat['content']}" for chat in chat_history])
    prompt = f"Summarize the following chat history:\n\n{chat_history_text}"
    messages = [{'role': 'system', 'content': 'You are very good at summarizing the chat between User and Assistant'}]
    messages.append({'role': 'user', 'content': prompt})
    summary = chat_groq(messages)
    return summary

# Main function to control the app
def main():
    st.set_page_config(page_title='DocLink LLM Bot 🌍')

    st.title("FinAdvisor Chatbot 🤖")
    with st.expander("Instructions to upload Text PDF/URL 📚"):
        st.write("1. Use the sidebar to upload a PDF or enter up to three URLs.")
        st.write("2. Click 'Process' to load documents and 'Create Vector Store' to build the knowledge base.")
        st.write("3. Submit your question to interact with the chatbot.")
        st.write("4. Generate a summary of the chat session if needed.")

    # Sidebar for document source selection
    st.sidebar.subheader("Choose document source: 🌐")
    option = st.sidebar.radio("Select one:", ("Upload PDF 📄", "Enter Web URLs 🌍"))

    if "docs" not in st.session_state:
        st.session_state.docs = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = ""
    if "chat_summary" not in st.session_state:
        st.session_state.chat_summary = ""
    if "previous_searches" not in st.session_state:
        st.session_state.previous_searches = []  # Store previous searches

    if option == "Upload PDF 📄":
        uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded_file is not None:
            if st.session_state.docs is None:
                with st.spinner("Loading documents... 🕒"):
                    docs = get_docs(uploaded_file)
                st.session_state.docs = docs

    elif option == "Enter Web URLs 🌍":
        url1 = st.sidebar.text_input("Enter URL 1", key="url1_input")
        url2 = st.sidebar.text_input("Enter URL 2 (optional)", key="url2_input")
        url3 = st.sidebar.text_input("Enter URL 3 (optional)", key="url3_input")
        urls = [url for url in [url1, url2, url3] if url]

        if st.sidebar.button('Process URLs 🌐'):
            if urls and st.session_state.docs is None:
                with st.spinner("Fetching and processing documents from URLs... 🌐"):
                    docs = get_docs_from_urls(urls)
                st.session_state.docs = docs

    if st.session_state.docs is not None:
        if st.sidebar.button('Create Vector Store 💾'):
            with st.spinner("Creating vector store... 🧠"):
                vectorstore = create_vector_store(st.session_state.docs)
            st.session_state.vectorstore = vectorstore

    if st.session_state.vectorstore is not None:
        def submit_with_doc():
            user_message = st.session_state.user_input
            if user_message:
                # Store the user input in the previous searches list
                if len(st.session_state.previous_searches) >= 5:
                    st.session_state.previous_searches.pop(0)
                st.session_state.previous_searches.append(user_message)

                retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                context = retriever.invoke(user_message)
                prompt = f'''
                Answer the user's question based on the latest input provided in the chat history.
                Ignore previous inputs unless they are directly related to the latest question.

                Context: {context}

                Chat History: {st.session_state.chat_history}

                Latest Question: {user_message}
                '''
                messages = [{'role': 'system', 'content': 'You are a very helpful assistant'}]
                messages.append({'role': 'user', 'content': prompt})
                try:
                    ai_response = chat_groq(messages)
                except Exception as e:
                    st.error(f"Error during chat_groq execution: {str(e)}")
                    ai_response = "An error occurred. Please try again."
                st.session_state.current_prompt = ai_response
                st.session_state.chat_history.append({'role': 'user', 'content': user_message})
                st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})
                st.session_state.user_input = ""

        st.text_area("Enter your question: 💬", key="user_input")
        if st.session_state.vectorstore is not None:
            st.button('Submit 💬', on_click=submit_with_doc)

    if st.session_state.current_prompt:
        st.write(st.session_state.current_prompt)

    if st.button('Generate Chat Summary 📝'):
        st.session_state.chat_summary = summarize_chat_history(st.session_state.chat_history)

    if st.session_state.chat_summary:
        with st.expander("Chat Summary 📝"):
            st.write(st.session_state.chat_summary)

    with st.expander("Recent Chat History 🕒"):
        recent_history = st.session_state.chat_history[-8:][::-1]
        for chat in recent_history:
            st.write(f"{chat['role'].capitalize()}: {chat['content']}")

    # Display Previous Searches
    with st.expander("Previous Searches 🔍"):
        for idx, search in enumerate(st.session_state.previous_searches[-5:], 1):
            st.write(f"{idx}. {search}")

if __name__ == "__main__":
    main()
