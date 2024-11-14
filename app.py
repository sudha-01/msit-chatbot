# app.py

import streamlit as st
import time
import json
from typing import List, Dict
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import io

# Download NLTK data
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')


class DocumentProcessor:
    def __init__(self, chunk_size=3):
        self.chunk_size = chunk_size

    def clean_text(self, text: str) -> str:
        """Clean the text content"""
        # Remove multiple spaces
        text = ' '.join(text.split())
        # Remove very long token strings (likely garbage from PDF)
        text = ' '.join(word for word in text.split() if len(word) < 30)
        return text

    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Load data from JSONL file"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        item = json.loads(line)
                        if all(key in item for key in ['url', 'type', 'content']):
                            data.append(item)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line: {line[:100]}...")
        except Exception as e:
            print(f"Error loading file: {str(e)}")
        return data

    def extract_text_from_jsonl(self, data: List[Dict]) -> List[Dict[str, str]]:
        """Extract text content from JSONL data with metadata"""
        processed_texts = []

        for item in data:
            if item['content'] and isinstance(item['content'], str):
                cleaned_text = self.clean_text(item['content'])
                if len(cleaned_text.split()) >= 10:  # Only keep substantial content
                    processed_texts.append({
                        'text': cleaned_text,
                        'url': item['url'],
                        'type': item['type']
                    })

        return processed_texts

    def create_chunks(self, texts: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Create chunks from texts while preserving metadata"""
        chunks = []

        for text_item in texts:
            # Split into sentences
            sentences = sent_tokenize(text_item['text'])

            # Create chunks of sentences
            for i in range(0, len(sentences), self.chunk_size):
                chunk_text = ' '.join(sentences[i:i + self.chunk_size])
                if len(chunk_text.split()) >= 10:  # Only keep chunks with at least 10 words
                    chunks.append({
                        'text': chunk_text,
                        'url': text_item['url'],
                        'type': text_item['type']
                    })

        return chunks


class RAGSystem:
    def __init__(self):
        # Initialize GPT-2 model and tokenizer
        self.llm = GPT2LMHeadModel.from_pretrained('gpt2')
        self.llm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize FAISS index
        self.vector_dimension = 384
        self.vector_index = faiss.IndexFlatL2(self.vector_dimension)

        # Storage for documents and metadata
        self.documents: List[Dict[str, str]] = []

    def add_documents(self, documents: List[Dict[str, str]]):
        """Add documents to the knowledge base"""
        # Store documents with metadata
        self.documents.extend(documents)

        # Create embeddings for the text content
        embeddings = self.embedding_model.encode([doc['text'] for doc in documents])

        # Add to FAISS index
        self.vector_index.add(np.array(embeddings).astype('float32'))

        return len(documents)

    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        """Retrieve top-k relevant documents for the query"""
        query_embedding = self.embedding_model.encode([query])

        distances, indices = self.vector_index.search(
            np.array(query_embedding).astype('float32'), k
        )

        return [self.documents[i] for i in indices[0]]

    def generate_response(self, query: str, context: List[Dict[str, str]]) -> Dict:
        """Generate response using GPT-2 with retrieved context"""
        # Combine context texts
        context_texts = [doc['text'] for doc in context]
        context_urls = [doc['url'] for doc in context]

        prompt = f"Context: {' '.join(context_texts)}\n\nQuestion: {query}\n\nAnswer:"

        inputs = self.llm_tokenizer(prompt, return_tensors="pt", padding=True)

        output_sequences = self.llm.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=self.llm_tokenizer.eos_token_id
        )

        response_text = self.llm_tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        response_text = response_text.split("Answer:")[-1].strip()

        return {
            'response': response_text,
            'sources': list(set(context_urls))  # Deduplicated source URLs
        }

    def process_query(self, query: str) -> Dict:
        """Main RAG pipeline: retrieve + generate"""
        relevant_chunks = self.retrieve_relevant_chunks(query)
        response = self.generate_response(query, relevant_chunks)
        return response


def get_custom_css():
    return """
    <style>
        .user-message {
            background-color: #DCF8C6;
            padding: 10px;
            border-radius: 10px;
            margin: 5px;
            float: right;
            clear: both;
            max-width: 70%;
        }
        .bot-message {
            background-color: #E8E8E8;
            padding: 10px;
            border-radius: 10px;
            margin: 5px;
            float: left;
            clear: both;
            max-width: 70%;
        }
        .sources {
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
        }
        .message-container {
            margin-bottom: 15px;
            overflow: hidden;
        }
        .stButton>button {
            width: 100%;
        }
    </style>
    """


def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False


def load_rag_system_from_upload(uploaded_file):
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            content = uploaded_file.getvalue().decode('utf-8')
            lines = content.strip().split('\n')
            data = [json.loads(line) for line in lines]

            # Initialize the system
            doc_processor = DocumentProcessor(chunk_size=3)
            rag_system = RAGSystem()

            # Process the data
            texts = doc_processor.extract_text_from_jsonl(data)
            chunks = doc_processor.create_chunks(texts)

            # Add documents to the system
            num_chunks = rag_system.add_documents(chunks)

            st.session_state.rag_system = rag_system
            st.session_state.file_uploaded = True
            st.success(f'Successfully loaded {num_chunks} chunks from the uploaded file!')

        except Exception as e:
            st.error(f'Error processing file: {str(e)}')
            st.session_state.file_uploaded = False
    else:
        st.session_state.file_uploaded = False


def main():
    st.set_page_config(page_title="College ChatBot", page_icon="🎓", layout="wide")
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    initialize_session_state()

    st.title("College ChatBot 🎓")

    # File upload section
    if not st.session_state.file_uploaded:
        st.write("Please upload your JSONL file to start chatting:")
        uploaded_file = st.file_uploader("Choose a JSONL file", type=['jsonl'])
        if uploaded_file is not None:
            load_rag_system_from_upload(uploaded_file)

    # Only show chat interface if file is uploaded
    if st.session_state.file_uploaded:
        # Create a two-column layout
        col1, col2 = st.columns([3, 1])

        with col1:
            # Chat interface
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.messages:
                    if message['role'] == 'user':
                        st.markdown(f"""
                            <div class="message-container">
                                <div class="user-message">{message['content']}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="message-container">
                                <div class="bot-message">
                                    {message['content']}
                                    <div class="sources">Sources: {', '.join(message['sources'])}</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

            # Input section
            with st.container():
                user_input = st.text_input("Type your message...", key="user_input")
                if st.button("Send", key="send_button") or user_input:
                    if user_input:
                        # Add user message
                        st.session_state.messages.append({
                            'role': 'user',
                            'content': user_input
                        })

                        # Get bot response
                        with st.spinner('Thinking...'):
                            response = st.session_state.rag_system.process_query(user_input)

                        # Add bot response
                        st.session_state.messages.append({
                            'role': 'assistant',
                            'content': response['response'],
                            'sources': response['sources']
                        })

                        # Clear input and rerun
                        st.experimental_rerun()

        with col2:
            # Sidebar for additional controls
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.experimental_rerun()

            if st.button("Upload New File"):
                st.session_state.file_uploaded = False
                st.session_state.rag_system = None
                st.session_state.messages = []
                st.experimental_rerun()


if __name__ == "__main__":
    main()
