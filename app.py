import streamlit as st
from utils import (
    load_tickets,
    chunk_text,
    create_vector_store,
    retrieve_similar_tickets,
    classify_ticket,
    detect_severity,
    generate_structured_response,
    recommend_sla,
    escalation_required
)

st.set_page_config(page_title="AI Ticket Automation System", layout="wide")

st.title("AI Customer Support Ticket Automation System")
st.markdown("Automated Ticket Triage, Severity Detection, SLA Recommendation & Resolution Generation")

@st.cache_resource
def initialize_system():
    tickets_data = load_tickets()
    chunks = chunk_text(tickets_data)
    index, embeddings = create_vector_store(chunks)
    return index, chunks

index, chunks = initialize_system()

st.subheader("Enter New Support Ticket")

ticket_input = st.text_area("Describe the issue:", height=150)

support_mode = st.radio(
    "Choose Support Mode:",
    ["AI Agent", "Human Agent"],
    horizontal=True
)

if ticket_input:

    st.divider()
    st.subheader("Ticket Analysis")

    # 1️⃣ Classification
    category = classify_ticket(ticket_input)

    # 2️⃣ Severity Detection
    severity = detect_severity(ticket_input)

    # 3️⃣ SLA & Escalation
    sla = recommend_sla(severity)
    escalate = escalation_required(severity)

    # ⚠ Smart recommendation
    if severity == "High":
        st.info("⚠ High severity detected. Human agent is recommended.")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Predicted Category:**", category)
        st.write("**Severity Level:**", severity)

    with col2:
        st.write("**Recommended SLA:**", sla)
        st.write("**Escalation Required:**", "Yes" if escalate else "No")

    st.divider()

    # Retrieval
    st.subheader("Similar Past Tickets Retrieved")
    similar_tickets = retrieve_similar_tickets(ticket_input, index, chunks)

    for ticket in similar_tickets:
        st.markdown(f"- {ticket}")

    st.divider()

    # Support handling
    if support_mode == "AI Agent":
        st.subheader("Automated Resolution Output")
        structured_response = generate_structured_response(ticket_input, similar_tickets)
        st.text_area("AI Generated Resolution", structured_response, height=300)

    else:
        st.subheader("Human Support Assigned")
        st.warning("This ticket has been assigned to a human support representative.")
        st.write("Recommended SLA:", sla)

        if escalate:
            st.error("High priority ticket. Escalated to senior support team.")

        st.success("Your ticket has been successfully queued. A support representative will contact you shortly.")