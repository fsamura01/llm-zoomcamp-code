import streamlit as st
import pandas as pd
from dataclasses import asdict
from assistant import create_assistant
from db_save import save_conversation
from db_query import get_conversations, get_stats
from db_feedback import save_feedback
from judge import evaluate_relevance
from db_query import get_relevance_stats, get_user_feedback_stats

assistant = create_assistant()
stats = get_stats()

tab1, tab2, tab3, tab4 = st.tabs(["💻 Chat", "📊 Monitoring Dashboard", "📝 Feedback Dashboard", "⚙️ Settings"])



with tab1:
    st.title("Course Assistant")
    records = get_conversations(limit=100)
    for record in records:
        st.write(f"**{record.prompt[:80]}...**")
        st.write(f"{record.answer[:200]}...")
        st.write(f"Time: {record.response_time:.2f}s | Cost: ${record.cost:.4f}")
        st.divider()

  

    user_input = st.text_input("Enter your question:")
    if st.button("Ask"):
        with st.spinner("Processing..."):
            answer = assistant.rag(user_input)
            st.success("Completed!")
            st.write(answer)

            record = assistant.last_call
            st.write(f"Response time: {record.response_time:.2f}s")
            st.write(f"Prompt tokens: {record.prompt_tokens}")
            st.write(f"Completion tokens: {record.completion_tokens}")
            st.write(f"Cost: ${record.cost:.4f}")

            conversation_id = save_conversation(record, user_input, "llm-zoomcamp")
            st.session_state.conversation_id = conversation_id

            relevance, explanation = evaluate_relevance(user_input, answer)
            save_feedback(conversation_id, "judge",
                relevance=relevance, explanation=explanation)
            st.write(f"Relevance: {relevance}")
            st.write(f"Explanation: {explanation}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("+1"):
                    cid = st.session_state.conversation_id
                    save_feedback(cid, "user", score=1)
                    st.write("Thanks!")

            with col2:
                if st.button("-1"):
                    cid = st.session_state.conversation_id
                    save_feedback(cid, "user", score=-1)
                    st.write("Thanks for the feedback!")
       
with tab2:
    st.title("Performance & Monitoring")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total conversations", stats.total)
    col2.metric("Avg response time", f"{stats.avg_response_time:.2f}s")
    col3.metric("Total cost", f"${stats.total_cost:.4f}")
    col4.metric("Avg tokens", f"{stats.avg_tokens:.0f}")

    records = get_conversations(limit=100)
    df = pd.DataFrame([asdict(r) for r in records])

    st.subheader("Cost over time")
    st.line_chart(df, x="timestamp", y="cost")

    st.subheader("Response time over time")
    st.line_chart(df, x="timestamp", y="response_time")

with tab3:
    st.subheader("Judge relevance")
    relevance = get_relevance_stats()
    st.bar_chart(relevance)
    st.subheader("User feedback")

    thumbs_up, thumbs_down = get_user_feedback_stats()
    col1, col2 = st.columns(2)
    col1.metric("Thumbs up", int(thumbs_up or 0))
    col2.metric("Thumbs down", int(thumbs_down or 0))