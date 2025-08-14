import os
import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from dotenv import load_dotenv

# Import the LLM handler functions you created
from llm_handlers import ask_openai, ask_groq

# === CONFIG ===

# Dynamically set paths based on the current Python file location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, 'Combined_FAISS', 'combined_index.bin')
METADATA_PATH = os.path.join(BASE_DIR, 'Combined_FAISS', 'combined_metadata.pkl')

TOP_K = 5

# Choose whether to get raw chunks or generate an LLM answer
USE_LLM_ANSWER = True  # Set False to just print top chunks, True to generate answer via LLM

# Choose which LLM provider to use: "openai" or "groq"
LLM_PROVIDER = "groq" 

# Load from environment variables
load_dotenv()  # take environment variables from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Workaround for OpenMP lib conflicts

###########################################################################################
### Suppress warnings from (UserWarning: 1Torch was not compiled with flash attention.) ###
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
###########################################################################################

# === Load FAISS index ===
print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

# === Load metadata ===
print("Loading metadata...")
with open(METADATA_PATH, 'rb') as f:
    metadata = pickle.load(f)

# === Load embedding model ===
print("Loading embedding model (BAAI/bge-base-en-v1.5)...")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def embed(text: str) -> np.ndarray:
    with torch.no_grad():
        inputs = tokenizer([text], return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        masked_hidden = last_hidden * attention_mask
        summed = masked_hidden.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        embedding = (summed / counts)
        normalized = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return normalized.cpu().numpy()

def search_faiss(query: str, k: int = TOP_K):
    query_vec = embed(query).astype('float32')
    scores, indices = index.search(query_vec, k)
    return [(metadata[i], float(scores[0][j])) for j, i in enumerate(indices[0]) if i < len(metadata)]

def format_context_chunks_with_citations(chunks):
    """
    Format each chunk with citation info prepended, to feed LLM.
    Example format for each chunk:
    [File: 104-10333-10001.pdf, Pages 106–107]
    <chunk text>
    """
    formatted_chunks = []
    for chunk in chunks:
        citation = f"[File: {chunk.get('file_name', 'Unknown')}, Pages {chunk.get('start_file', '?')}–{chunk.get('end_file', '?')}]"
        text = chunk.get('chunk', '')
        formatted_chunks.append(f"{citation}\n{text}")
    return "\n\n".join(formatted_chunks)

def aggregate_citations(chunks):
    """
    Collect unique citations from metadata chunks in a consistent formatted string.
    """
    seen = set()
    citations = []
    for chunk in chunks:
        file_name = chunk.get('file_name', 'Unknown')
        page_start = chunk.get('start_file', '?')
        page_end = chunk.get('end_file', '?')
        citation = f"[{file_name}] (Pages {page_start}–{page_end})"
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return "Sources: " + ", ".join(citations) if citations else "Sources: None"

def answer_with_llm(question: str, top_chunks: list):
    # Format chunks with citations for LLM context
    cite_text = True

    if cite_text:
        context_text = format_context_chunks_with_citations(top_chunks)  # Format with citations
    else: 
        context_text = "\n\n".join(chunk.get('chunk', '') for chunk in top_chunks) # format without citations

    # Create prompt for LLM including system instructions
    # prompt = (
    #     "You are a helpful research assistant. Your job is to answer the user's question strictly based "
    #     "on the provided context below. Do not invent information. \n"
    #     "Provide a clear, concise answer, and cite **all sources only at the end** of your answer, "
    #     "in the format: Sources: [FileName] (Pages X-Y). "
    #     "Do NOT include citations after each paragraph or sentence. "
    #     "If the answer cannot be found in the context, say "
    #     "'The information is not available in the provided documents.'\n\n"
    #     f"Context:\n{context_text}\n\n"
    #     f"Question: {question}"
    # )

    # Create prompt for LLM including system instructions (without citations in the body)
    
    prompt = (
        "You are a helpful research assistant with access to declassified government documents. "
        "Your job is to answer the user's question using **only** the provided context. "
        "Do not make assumptions or fabricate information.\n\n"
        "Instructions:\n"
        "- Give a clear, concise answer to the question.\n"
        "- If the answer is not found in the context, reply: "
        "'The information is not available in the provided documents.'\n"
        "- Do not cite sources or list filenames in your answer.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}"
    )


    # print(f"\n\n\nLLM Prompt:\n{prompt}\n\n\n") # Uncomment to debug prompt

    if LLM_PROVIDER == "openai":
        raw_answer = ask_openai(question, top_chunks, OPENAI_API_KEY, prompt_override=prompt)
    elif LLM_PROVIDER == "groq":
        raw_answer = ask_groq(question, top_chunks, GROQ_API_KEY, prompt_override=prompt)
    else:
        raw_answer = "[Error] Unsupported LLM provider."

    # Aggregate citations from chunks metadata
    #citations = aggregate_citations(top_chunks) # This will be appended to the answer (Does not need to be in the prompt if the LLM is providing already)
    citations = ""
    
    # Append citations explicitly if not already present
    if citations not in raw_answer:
        return f"{raw_answer}\n\n{citations}"
    return raw_answer

def main():
    print("\nJFK QA Chatbot Ready! Ask your question below.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        top_results = search_faiss(question)
        
        if not top_results:
            print("No relevant results found.")
            continue
        
        if USE_LLM_ANSWER:
            # Use LLM to generate answer from top chunks
            answer = answer_with_llm(question, [meta for meta, _ in top_results])
            print("\nAI Answer:")
            print(answer)
        else:
            # Just print the top matching chunks as before
            print("\nTop Matches:")
            for i, (meta, score) in enumerate(top_results, 1):
                print(f"\n[{i}] (score: {score:.4f}) Pages {meta['start_file']}–{meta['end_file']} — File: {meta['file_name']}")
                print(meta['chunk'][:500] + '...')
        
        print("\n" + "-" * 80)

if __name__ == "__main__":
    main()
