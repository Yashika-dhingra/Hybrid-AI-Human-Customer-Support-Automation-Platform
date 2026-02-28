import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM for reasoning & response generation
generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=256
)

def load_tickets(file_path="tickets.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data

def chunk_text(text):
    tickets = text.split("Ticket ")
    chunks = []

    for t in tickets:
        if t.strip():
            chunks.append("Ticket " + t.strip())

    return chunks

def create_vector_store(chunks):
    embeddings = embedding_model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    return index, embeddings

def retrieve_similar_tickets(query, index, chunks, k=1):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(
        np.array(query_embedding).astype("float32"), k
    )

    results = [chunks[i] for i in indices[0]]
    return results

def classify_ticket(ticket_text):
    text = ticket_text.lower()

    if "payment" in text or "checkout" in text:
        return "Payment"
    elif "tracking" in text:
        return "Tracking"
    elif "invoice" in text or "billing" in text:
        return "Billing"
    elif "outage" in text or "system" in text:
        return "System Failure"
    elif "damaged" in text or "delivery" in text:
        return "Delivery Issue"
    elif "shipment" in text or "warehouse" in text:
        return "Logistics"
    else:
        return "Logistics"

def detect_severity(ticket_text):
    text = ticket_text.lower()

    if "system outage" in text or "multiple shipments" in text:
        return "High"
    elif "delayed" in text or "failed" in text:
        return "Medium"
    else:
        return "Low"
    
def generate_structured_response(ticket_text, retrieved_context):
    # Extract resolution from first similar ticket
    first_ticket = retrieved_context[0]

    resolution_line = ""
    for line in first_ticket.split("\n"):
        if "Resolution:" in line:
            resolution_line = line.replace("Resolution:", "").strip()

    if not resolution_line:
        resolution_line = "Investigate the issue and take appropriate action."

    # Structured deterministic output
    output = f"""
Root Cause:
Issue similar to previously recorded case.

Resolution Steps:
- {resolution_line}
- Verify issue resolution
- Notify customer

Customer Response:
We apologize for the inconvenience caused. Our team is currently addressing the issue and will keep you updated.
"""

    return output.strip()

def recommend_sla(severity):
    if severity == "High":
        return "Immediate attention (Within 1 hour)"
    elif severity == "Medium":
        return "Resolve within 4-8 hours"
    else:
        return "Resolve within 24 hours"
    
def escalation_required(severity):
    return severity == "High"