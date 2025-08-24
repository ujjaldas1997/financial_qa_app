import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# --- Load Models and Data ---
@st.cache_resource
def load_models():
    # Load Sentence Transformer model
    embedding_model = SentenceTransformer('embedding_model')

    # Load FAISS index
    index_dense = faiss.read_index('faiss_index.bin')

    # Load BM25 tokenized corpus
    with open('bm25_tokenized_corpus.json', 'r') as f:
        bm25_tokenized_corpus = json.load(f)
    index_sparse = BM25Okapi(bm25_tokenized_corpus)

    # Load CrossEncoder model
    cross_encoder = CrossEncoder('cross_encoder_model')

    # Load Fine-tuned Flan-T5 model and tokenizer
    tokenizer_flan = AutoTokenizer.from_pretrained('finetuned_model_flan_t5')
    model_flan = AutoModelForSeq2SeqLM.from_pretrained('finetuned_model_flan_t5')
    fine_tuned_model_pipeline = pipeline('text2text-generation', model=model_flan, tokenizer=tokenizer_flan)


    # Load chunk data (assuming it's needed for retrieval, regenerate or load as needed)
    # For this example, let's recreate a dummy chunk_data or load from a saved file if available
    # In a real application, you'd save and load this.
    # For now, we'll use a simplified representation based on the loaded BM25 corpus
    chunk_data = [{"text": " ".join(tokens)} for tokens in bm25_tokenized_corpus]


    return embedding_model, index_dense, index_sparse, cross_encoder, fine_tuned_model_pipeline, chunk_data

embedding_model, index_dense, index_sparse, cross_encoder, fine_tuned_model_pipeline, chunk_data = load_models()

# --- RAG Functions ---
def hybrid_retrieval(query, top_n=3):
    processed_query = query.lower()

    # Dense retrieval
    query_embedding = embedding_model.encode([processed_query])
    distances, indices_dense = index_dense.search(np.array(query_embedding).astype('float32'), top_n)

    # Sparse retrieval
    tokenized_query = processed_query.split(" ")
    scores_sparse = index_sparse.get_scores(tokenized_query)
    indices_sparse = np.argsort(scores_sparse)[::-1][:top_n]

    # Combine results
    combined_indices = sorted(list(set(indices_dense[0]) | set(indices_sparse)))
    return [chunk_data[i] for i in combined_indices]

def rerank(query, retrieved_docs):
    if not retrieved_docs:
        return []
    pairs = [[query, doc['text']] for doc in retrieved_docs]
    scores = cross_encoder.predict(pairs)
    reranked_indices = np.argsort(scores)[::-1]
    return [retrieved_docs[i] for i in reranked_indices]

def generate_response_rag(query):
    retrieved_docs = hybrid_retrieval(query)
    reranked_docs = rerank(query, retrieved_docs)
    context = "\n".join([doc['text'] for doc in reranked_docs[:2]]) # Use top 2 docs

    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

    # Use a pipeline for the generator model
    generator = pipeline('text2text-generation', model='google/flan-t5-small') # Using base Flan-T5 as generator
    response = generator(prompt, max_length=100)
    return response[0]['generated_text'].strip()

# --- Fine-Tuned Model Function ---
def generate_response_finetuned(query):
    prompt = f"Question: {query}"
    response = fine_tuned_model_pipeline(prompt, max_length=100)[0]['generated_text']
    return response.strip()

# --- Guardrails ---
def input_guardrail_rag(query):
    financial_keywords = ['revenue', 'profit', 'assets', 'liabilities', 'equity', 'financial', 'company', 'corp']
    if any(keyword in query.lower() for keyword in financial_keywords):
        return True
    return False

def output_guardrail_ft(response):
    non_committal_phrases = ["i don't know", "not available", "cannot answer", "no information"]
    if any(phrase in response.lower() for phrase in non_committal_phrases):
        return "The information is not available in the provided context."
    return response

# --- Streamlit App ---
st.title("Financial QA System: RAG vs Fine-Tuning")

st.write("Ask a question about the financial report to see how the RAG and Fine-Tuned models respond.")

query = st.text_input("Enter your question:")

if query:
    st.subheader("RAG Model Response:")
    if input_guardrail_rag(query):
        rag_response = generate_response_rag(query)
        st.write(rag_response)
    else:
        st.write("This question does not appear to be related to finance.")

    st.subheader("Fine-Tuned Model Response:")
    finetuned_response_raw = generate_response_finetuned(query)
    finetuned_response = output_guardrail_ft(finetuned_response_raw)
    st.write(finetuned_response)
