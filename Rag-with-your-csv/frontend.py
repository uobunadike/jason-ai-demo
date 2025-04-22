

import os
import streamlit as st
from model import run

# Configure UI
st.set_page_config(page_title="Inventory Analytics", layout="wide")

# Session state initialization
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar controls
with st.sidebar:
    st.title("Analytics Settings")
    model_type = st.radio(
        "Model Type", 
        ["ollama", "openai"], 
        index=0,
        help="Choose between local (Ollama) or cloud (OpenAI) models"
    )
    model_name = st.selectbox(
        "Model Version",
        options=["llama3.1", "mistral"] if model_type == "ollama" else ["gpt-4-turbo", "gpt-3.5-turbo"],
        index=0
    )

# Main chat interface
col1, col2 = st.columns([3, 1])
with col1:
    st.header("📊 Real-time Inventory Analysis")
    
    # Display chat history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
    
    # User input
    if prompt := st.chat_input("Ask about stock levels or sales trends..."):
        # Add user message
        st.session_state.history.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner("🧠 Analyzing inventory data..."):
            try:
                response = run(prompt, model_type, model_name)
                response += f"\n\n*Generated using {model_name}*"
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"
        
        # Add assistant response
        st.session_state.history.append({"role": "assistant", "content": response})
        st.rerun()

# Quick analysis panel
with col2:
    st.header("🚀 Quick Insights")
    st.markdown("""
    **Common Queries:**
    - Current stockout rate for SKU-202
    - Inventory turnover last month
    - Recent sales trends for Widget E
    - COGS calculation for Q1 2025
    """)
    
    st.divider()
    
    st.markdown("**Key Metrics**")
    st.metric("Total SKUs Tracked", "287", "+12% vs last month")
    st.metric("Avg Stockout Rate", "8.2%", "-2.1% MoM")
    st.metric("Inventory Turnover", "5.8x", "▲ 0.4x")

# Cache management (optional)
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cache cleared successfully!")
