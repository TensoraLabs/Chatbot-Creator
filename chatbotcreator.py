import streamlit as st
import requests
import json
from typing import Dict, List
import os

# Page configuration
st.set_page_config(
    page_title="AI Chatbot Creator",
    page_icon="🤖",
    layout="wide"
)

class LLMChatbot:
    def __init__(self):
        self.api_configs = {
            "Groq (Free)": {
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "models": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
                "free_tier": True
            },
            "OpenAI": {
                "url": "https://api.openai.com/v1/chat/completions", 
                "models": ["gpt-3.5-turbo", "gpt-4o-mini"],
                "free_tier": False
            },
            "OpenRouter (Free Models)": {
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "models": ["microsoft/dialoGPT-medium", "huggingface/CodeBERTa-small-v1"],
                "free_tier": True
            },
            "Cohere": {
                "url": "https://api.cohere.ai/v1/chat",
                "models": ["command-light", "command"],
                "free_tier": True
            }
        }
    
    def test_api_connection(self, api_name: str, api_key: str) -> bool:
        try:
            if api_name == "Groq (Free)":
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                response = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
                return response.status_code == 200
            elif api_name == "OpenAI":
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                response = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=10)
                return response.status_code == 200
            elif api_name == "Cohere":
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                data = {"message": "test", "model": "command-light"}
                response = requests.post("https://api.cohere.ai/v1/chat", headers=headers, json=data, timeout=10)
                return response.status_code in [200, 400]
            return False
        except:
            return False
    
    def chat_completion(self, api_name: str, api_key: str, model: str, messages: List[Dict], system_context: str = "") -> str:
        try:
            if api_name == "Groq (Free)" or api_name == "OpenAI":
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                formatted_messages = []
                if system_context:
                    formatted_messages.append({"role": "system", "content": system_context})
                formatted_messages.extend(messages)
                
                data = {
                    "model": model,
                    "messages": formatted_messages,
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
                
                response = requests.post(self.api_configs[api_name]["url"], headers=headers, json=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    return f"API Error {response.status_code}: {response.text}"
            
            elif api_name == "Cohere":
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                # Format conversation for Cohere
                conversation_history = []
                if system_context:
                    conversation_history.append({"role": "SYSTEM", "message": system_context})
                
                for msg in messages[:-1]:
                    role = "USER" if msg["role"] == "user" else "CHATBOT"
                    conversation_history.append({"role": role, "message": msg["content"]})
                
                data = {
                    "message": messages[-1]["content"],
                    "model": model,
                    "chat_history": conversation_history,
                    "temperature": 0.7
                }
                
                response = requests.post("https://api.cohere.ai/v1/chat", headers=headers, json=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    return result["text"]
                else:
                    return f"API Error {response.status_code}: {response.text}"
            
            elif api_name == "OpenRouter (Free Models)":
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://streamlit.io",
                    "X-Title": "Streamlit Chatbot"
                }
                
                formatted_messages = []
                if system_context:
                    formatted_messages.append({"role": "system", "content": system_context})
                formatted_messages.extend(messages)
                
                data = {
                    "model": model,
                    "messages": formatted_messages,
                    "temperature": 0.7,
                    "max_tokens": 500
                }
                
                response = requests.post(self.api_configs[api_name]["url"], headers=headers, json=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    return f"API Error {response.status_code}: {response.text}"
                    
        except Exception as e:
            return f"Error: {str(e)}"
        
llm_chatbot = LLMChatbot()

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chatbot_context" not in st.session_state:
    st.session_state.chatbot_context = ""
if "chatbot_created" not in st.session_state:
    st.session_state.chatbot_created = False
if "api_configured" not in st.session_state:
    st.session_state.api_configured = False
if "selected_api" not in st.session_state:
    st.session_state.selected_api = "Groq (Free)"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

st.title("AI Chatbot Creator")
st.write("Create custom AI chatbots using real LLM models")

# Sidebar - API Configuration
with st.sidebar:
    st.header("🔧 API Configuration")
    
    # API Selection
    selected_api = st.selectbox(
        "Choose LLM Provider:",
        list(llm_chatbot.api_configs.keys()),
        index=list(llm_chatbot.api_configs.keys()).index(st.session_state.selected_api)
    )
    st.session_state.selected_api = selected_api
    
    # Show API info
    api_info = llm_chatbot.api_configs[selected_api]
    if api_info["free_tier"]:
        st.success("✅ This provider offers free tier!")
    else:
        st.warning("⚠️ This provider requires paid API access")
    
    # API Key input
    if selected_api == "Groq (Free)":
        st.info("Get free API key from: https://console.groq.com/keys")
    elif selected_api == "OpenAI":
        st.info("Get API key from: https://platform.openai.com/api-keys")
    elif selected_api == "Cohere":
        st.info("Get free API key from: https://dashboard.cohere.ai/api-keys")
    elif selected_api == "OpenRouter (Free Models)":
        st.info("Get free credits from: https://openrouter.ai/keys")
    
    api_key = st.text_input(
        "API Key:",
        type="password",
        value=st.session_state.api_key,
        help="Enter your API key for the selected provider"
    )
    st.session_state.api_key = api_key
    
    # Model selection
    if api_key:
        available_models = api_info["models"]
        if available_models:
            selected_model = st.selectbox(
                "Select Model:",
                available_models,
                help="Choose the model for your chatbot"
            )
            st.session_state.selected_model = selected_model
    
    # Test API connection
    if st.button("🔍 Test API Connection"):
        if api_key:
            with st.spinner("Testing connection..."):
                if llm_chatbot.test_api_connection(selected_api, api_key):
                    st.success("✅ API connection successful!")
                    st.session_state.api_configured = True
                else:
                    st.error("❌ API connection failed. Check your API key.")
                    st.session_state.api_configured = False
        else:
            st.error("Please enter an API key first.")
    
    st.divider()
    
    # Chatbot Configuration
    st.header("🎭 Chatbot Setup")
    
    if st.session_state.api_configured:
        # Context input
        context_input = st.text_area(
            "Chatbot Context:",
            value=st.session_state.chatbot_context,
            height=150,
            placeholder="Define your chatbot's personality, role, and behavior...",
            help="This will be the system prompt that defines your chatbot's behavior"
        )
        
        # Example contexts
        st.write("**Quick Examples:**")
        examples = {
            "Cooking Assistant": "You are a professional chef and cooking instructor. Help users with recipes, cooking techniques, and meal planning. Be enthusiastic and encouraging.",
            "Study Tutor": "You are a patient and knowledgeable tutor. Help students understand concepts, provide explanations, and create study strategies.",
            "Fitness Coach": "You are a certified fitness trainer. Provide workout advice, motivation, and healthy lifestyle tips. Be encouraging and safety-focused.",
            "Creative Writer": "You are a creative writing mentor. Help with story ideas, character development, and writing techniques. Be inspiring and constructive."
        }
        
        for name, context in examples.items():
            if st.button(f"Use {name}", key=f"example_{name}"):
                st.session_state.chatbot_context = context
                st.rerun()
        
        if st.button("🚀 Create Chatbot", type="primary"):
            if context_input.strip():
                st.session_state.chatbot_context = context_input.strip()
                st.session_state.chatbot_created = True
                st.session_state.chat_history = []
                st.success("Chatbot created successfully!")
                st.rerun()
            else:
                st.error("Please provide a context for your chatbot.")
        
        # Reset options
        if st.session_state.chatbot_created:
            if st.button("🔄 Reset Chat"):
                st.session_state.chat_history = []
                st.rerun()
            
            if st.button("🗑️ Delete Chatbot"):
                st.session_state.chatbot_created = False
                st.session_state.chatbot_context = ""
                st.session_state.chat_history = []
                st.rerun()
    
    else:
        st.info("Please configure and test your API connection first.")

# Main Chat Interface
if not st.session_state.api_configured:
    st.info("🔧 Please configure your API connection in the sidebar to get started.")
    
    # Show setup instructions
    st.subheader("🚀 Quick Setup Guide:")
    
    st.write("**Step 1: Choose a Provider**")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Recommended (Free):**")
        st.write("• Groq - Fast inference, free tier")
        st.write("• Cohere - Good for conversations")
    with col2:
        st.write("**Premium Options:**")
        st.write("• OpenAI - GPT models")
        st.write("• OpenRouter - Multiple models")
    
    st.write("**Step 2: Get API Key**")
    st.write("Sign up at your chosen provider and get a free API key")
    
    st.write("**Step 3: Test Connection**")
    st.write("Enter your API key and test the connection")
    
elif not st.session_state.chatbot_created:
    st.info("🎭 Please create your chatbot in the sidebar to start chatting.")
    
    # Show current API status
    st.success(f"✅ Connected to {st.session_state.selected_api}")
    if st.session_state.selected_model:
        st.info(f"Model: {st.session_state.selected_model}")

else:
    # Show chatbot info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("💬 Chat with Your AI Assistant")
    with col2:
        st.write(f"**Provider:** {st.session_state.selected_api}")
        st.write(f"**Model:** {st.session_state.selected_model}")
    
    # Show context
    with st.expander("🤖 Current Chatbot Context"):
        st.write(st.session_state.chatbot_context)
    
    # Chat display
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write("Hello! I'm your custom AI assistant. How can I help you today?")
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        

        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = llm_chatbot.chat_completion(
                    st.session_state.selected_api,
                    st.session_state.api_key,
                    st.session_state.selected_model,
                    st.session_state.chat_history,
                    st.session_state.chatbot_context
                )
            
            st.write(response)
            

            st.session_state.chat_history.append({"role": "assistant", "content": response})


st.divider()
st.write("Built with Streamlit • Connect your own LLM API for real AI conversations")
