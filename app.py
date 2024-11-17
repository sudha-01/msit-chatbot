import streamlit as st
import json
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict
import re

# Initialize Streamlit session state to maintain conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []


class DocumentProcessor:
    """Class to handle document processing and chunking"""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean the text by removing extra whitespace and special characters"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    @staticmethod
    def create_semantic_chunks(documents: List[Dict], max_chunk_size: int = 512) -> List[Dict]:
        """Create semantic chunks from documents"""
        chunks = []

        for doc in documents:
            content = doc['content']
            # Split content into sentences (basic approach)
            sentences = re.split(r'(?<=[.!?])\s+', content)

            current_chunk = ""
            current_url = doc['url']
            current_type = doc['type']

            for sentence in sentences:
                # If adding new sentence exceeds max_chunk_size, save current chunk and start new one
                if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                    chunks.append({
                        'url': current_url,
                        'type': current_type,
                        'content': DocumentProcessor.clean_text(current_chunk)
                    })
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence

            # Add the last chunk if it exists
            if current_chunk:
                chunks.append({
                    'url': current_url,
                    'type': current_type,
                    'content': DocumentProcessor.clean_text(current_chunk)
                })

        return chunks

    @staticmethod
    def load_jsonl(file) -> List[Dict]:
        """Load and parse JSONL file"""
        documents = []
        content = file.getvalue().decode('utf-8')
        for line in content.strip().split('\n'):
            try:
                doc = json.loads(line)
                if all(k in doc for k in ['url', 'type', 'content']):  # Validate required fields
                    documents.append(doc)
                else:
                    st.warning(f"Skipping invalid document: missing required fields")
            except json.JSONDecodeError:
                st.warning(f"Skipping invalid JSON line")
        return documents


class VectorStore:
    """Class to handle vector storage and retrieval"""

    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []

    def create_embeddings(self, chunks: List[Dict]):
        """Create embeddings for chunks and build FAISS index"""
        embeddings = []
        self.documents = chunks

        # Show progress bar for embedding creation
        progress_bar = st.progress(0)
        for idx, chunk in enumerate(chunks):
            embedding = self.model.encode(chunk['content'], convert_to_tensor=True)
            embeddings.append(embedding.cpu().numpy())
            progress_bar.progress((idx + 1) / len(chunks))

        embeddings_array = np.array(embeddings).astype('float32')

        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_array)
        progress_bar.empty()  # Remove progress bar after completion

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for most relevant chunks given a query"""
        query_vector = self.model.encode(query, convert_to_tensor=True)
        query_vector_np = query_vector.cpu().numpy().reshape(1, -1).astype('float32')

        # Search in FAISS index
        distances, indices = self.index.search(query_vector_np, k)

        # Return relevant documents
        return [self.documents[i] for i in indices[0]]


class LLMClient:
    """Class to handle interactions with the RunPod LLM endpoint"""

    def __init__(self, url: str):
        self.url = url
        self.headers = {'Content-Type': 'application/json'}

    def get_response(self, prompt: str, max_tokens: int = 200) -> str:
        """Get response from LLM"""
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 10
        }

        response = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            verify=False
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['text']
        else:
            raise Exception(f"Failed to get response: {response.status_code}")


def main():
    st.title("RAG Chatbot")

    # Initialize components
    llm_client = LLMClient("https://ke38c4ecupwr0t-5000.proxy.runpod.net/v1/completions")
    vector_store = VectorStore()

    # Sidebar for file upload and system status
    with st.sidebar:
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload your JSONL data", type=['jsonl'])

        if uploaded_file:
            st.write("File Upload Status:")
            # Load and process documents
            with st.spinner('Loading JSONL file...'):
                documents = DocumentProcessor.load_jsonl(uploaded_file)
                st.success(f"✓ Loaded {len(documents)} documents")

            processor = DocumentProcessor()
            with st.spinner('Creating chunks...'):
                chunks = processor.create_semantic_chunks(documents)
                st.success(f"✓ Created {len(chunks)} chunks")

            # Create embeddings and index
            with st.spinner('Creating embeddings...'):
                vector_store.create_embeddings(chunks)
                st.success("✓ Created embeddings")

    # Main chat interface
    if vector_store.index is not None:  # Only show chat if index is created
        st.markdown("""
        ### Chat Interface
        Ask questions about your uploaded documents. The system will:
        1. Find relevant content from your documents
        2. Generate a response based on the found content
        """)

        # Chat input
        user_input = st.text_input("Your question:", key="user_input")

        if user_input:
            # Search relevant documents
            relevant_docs = vector_store.search(user_input)

            # Create context from relevant documents
            context = "\n".join([doc['content'] for doc in relevant_docs])

            # Create prompt for LLM
            prompt = f"""Based on the following context, please answer the question. 
            If the answer cannot be found in the context, say 'I don't have enough information to answer that.'

            Context:
            {context}

            Question: {user_input}

            Answer:"""

            # Get response from LLM
            with st.spinner('Thinking...'):
                try:
                    response = llm_client.get_response(prompt)

                    # Add to conversation history
                    st.session_state.conversation_history.append(("User", user_input))
                    st.session_state.conversation_history.append(("Assistant", response))
                except Exception as e:
                    st.error(f"Error getting response: {str(e)}")

        # Display conversation history
        if st.session_state.conversation_history:
            st.subheader("Conversation History")
            for role, message in st.session_state.conversation_history:
                if role == "User":
                    st.markdown(f"🧑 **You:** {message}")
                else:
                    st.markdown(f"🤖 **Assistant:** {message}")
    else:
        st.info("👈 Please upload a JSONL file in the sidebar to start chatting!")


if __name__ == "__main__":
    main()
