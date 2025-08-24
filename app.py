import streamlit as st
import json
import os
import time
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from datasets import Dataset
import re
from fpdf import FPDF # Import FPDF

# --- Model Training and Setup ---
# This part will now be included in the app.py

def create_dummy_pdf(pdf_path="dummy_financial_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size = 12)
    pdf.multi_cell(0, 10, txt="""
Financial Report for XYZ Corp - 2023

Income Statement:
Revenue: $4.13 billion
Cost of Revenue: $1.5 billion
Gross Profit: $2.63 billion

Balance Sheet:
Total Assets: $10 billion
Total Liabilities: $5 billion
Shareholder Equity: $5 billion

---

Financial Report for XYZ Corp - 2022

Income Statement:
Revenue: $3.5 billion
Cost of Revenue: $1.2 billion
Gross Profit: $2.3 billion

Balance Sheet:
Total Assets: $8 billion
Total Liabilities: $4 billion
Shareholder Equity: $4 billion
""")
    pdf.output(pdf_path)


def extract_text_from_pdf(pdf_path):
    # This is a dummy function, in a real app you would use a PDF parsing library
    return """
Financial Report for XYZ Corp - 2023. Income Statement: Revenue: $4.13 billion, Cost of Revenue: $1.5 billion, Gross Profit: $2.63 billion. Balance Sheet: Total Assets: $10 billion, Total Liabilities: $5 billion, Shareholder Equity: $5 billion.
Financial Report for XYZ Corp - 2022. Income Statement: Revenue: $3.5 billion, Cost of Revenue: $1.2 billion, Gross Profit: $2.3 billion. Balance Sheet: Total Assets: $8 billion, Total Liabilities: $4 billion, Shareholder Equity: $4 billion.
"""

def clean_text(text):
    text = text.replace('\n', ' ').strip()
    return text

def split_into_chunks(text, chunk_size, overlap=20):
    tokenizer_flan = AutoTokenizer.from_pretrained('google/flan-t5-small')
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(tokenizer_flan.encode(current_chunk + sentence)) < chunk_size:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

@st.cache_resource # Cache the models and indices
def setup_models_and_data():
    # Create dummy PDF
    create_dummy_pdf()

    # Extract and clean text
    financial_text_raw = extract_text_from_pdf('dummy_financial_report.pdf')
    financial_text = clean_text(financial_text_raw)

    # Chunking
    chunks = split_into_chunks(financial_text, 100)
    chunk_data = [{"id": f"chunk_{i}", "text": chunk} for i, chunk in enumerate(chunks)]

    # Embed chunks
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode([doc['text'] for doc in chunk_data])

    # Dense vector store (FAISS)
    dimension = embeddings.shape[1]
    index_dense = faiss.IndexFlatL2(dimension)
    index_dense.add(np.array(embeddings).astype('float32'))

    # Sparse index (BM25)
    tokenized_corpus = [doc['text'].split(" ") for doc in chunk_data]
    index_sparse = BM25Okapi(tokenized_corpus)

    # Cross-encoder for re-ranking
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Load and fine-tune Flan-T5
    tokenizer_ft = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model_ft = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    # Construct Q/A pairs for fine-tuning
    qa_pairs = [
        {"question": "What was the company’s revenue in 2023?", "answer": "The company’s revenue in 2023 was $4.13 billion."},
        {"question": "What was the gross profit in 2022?", "answer": "The gross profit in 2022 was $2.3 billion."},
        {"question": "What were the total assets in 2023?", "answer": "The total assets in 2023 were $10 billion."},
        {"question": "What was the shareholder equity in 2022?", "answer": "The shareholder equity in 2022 was $4 billion."},
        {"question": "What was the cost of revenue in 2023?", "answer": "The cost of revenue in 2023 was $1.5 billion."},
        {"question": "What were the total liabilities in 2022?", "answer": "The total liabilities in 2022 was $4 billion."},
         {"question": "How much revenue did the company make in 2022?", "answer": "The company's revenue in 2022 was $3.5 billion."},
        {"question": "What was the gross profit for 2023?", "answer": "In 2023, the gross profit was $2.63 billion."},
        {"question": "Compare total assets between 2022 and 2023.", "answer": "Total assets were $8 billion in 2022 and $10 billion in 2023."},
        {"question": "What was the shareholder equity in 2023?", "answer": "Shareholder equity in 2023 was $5 billion."},
        {"question": "Did total liabilities increase from 2022 to 2023?", "answer": "Yes, total liabilities increased from $4 billion in 2022 to $5 billion in 2023."},
        {"question": "Calculate the profit margin for 2023 (Gross Profit / Revenue).", "answer": "The gross profit margin for 2023 was approximately 63.7% ($2.63 billion / $4.13 billion)."}
    ]

    ft_dataset = Dataset.from_list(qa_pairs)
    def format_dataset(example):
        return {'input_text': f"Question: {example['question']}", 'target_text': example['answer']}
    ft_dataset = ft_dataset.map(format_dataset)

    def tokenize_function(examples):
        inputs = tokenizer_ft(examples['input_text'], padding='max_length', truncation=True, return_tensors='pt')
        targets = tokenizer_ft(examples['target_text'], padding='max_length', truncation=True, return_tensors='pt')
        inputs['labels'] = targets['input_ids']
        return inputs
    tokenized_dataset = ft_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer_ft, model=model_ft)

    training_args = TrainingArguments(
        output_dir="./finetuned_model_flan_t5_app",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        save_steps=100,
        save_total_limit=2,
        learning_rate=3e-4,
        logging_dir='./logs_app',
        report_to='none'
    )

    trainer = Trainer(
        model=model_ft,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("./finetuned_model_flan_t5_app")

    fine_tuned_model_pipeline = pipeline('text2text-generation', model='./finetuned_model_flan_t5_app', tokenizer=tokenizer_ft)

    return chunk_data, embedding_model, index_dense, index_sparse, cross_encoder, fine_tuned_model_pipeline

chunk_data, embedding_model, index_dense, index_sparse, cross_encoder, fine_tuned_model_pipeline = setup_models_and_data()

# --- Retrieval and Generation Functions ---

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
    pairs = [(query, doc['text']) for doc in retrieved_docs]
    scores = cross_encoder.predict(pairs)
    reranked_indices = np.argsort(scores)[::-1]
    return [retrieved_docs[i] for i in reranked_indices]

def generate_response_rag(query):
    retrieved_docs = hybrid_retrieval(query)
    reranked_docs = rerank(query, retrieved_docs)
    context = "\n".join([doc['text'] for doc in reranked_docs[:2]])

    generator = pipeline('text2text-generation', model='google/flan-t5-small') # Load RAG generator here
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

    response = generator(prompt, max_length=100)
    return response[0]['generated_text'].strip()

def generate_response_finetuned(query):
    prompt = f"Question: {query}"
    response = fine_tuned_model_pipeline(prompt, max_length=100)[0]['generated_text']
    return response

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

st.title("Comparative Financial QA System")

st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Choose Model", ["RAG", "Fine-Tuned"])

st.header(f"{model_choice} Model")

query = st.text_input("Enter your financial question:")

if query:
    if model_choice == "RAG":
        if input_guardrail_rag(query):
            with st.spinner("Generating RAG response..."):
                start_time = time.time()
                response = generate_response_rag(query)
                end_time = time.time()
                st.write("Response:", response)
                st.write(f"Time taken: {end_time - start_time:.2f} seconds")
        else:
            st.warning("This question does not appear to be related to finance.")
            st.write("Response: Data not in scope")

    elif model_choice == "Fine-Tuned":
        with st.spinner("Generating Fine-Tuned response..."):
            start_time = time.time()
            response = generate_response_finetuned(query)
            response = output_guardrail_ft(response)
            end_time = time.time()
            st.write("Response:", response)
            st.write(f"Time taken: {end_time - start_time:.2f} seconds")
