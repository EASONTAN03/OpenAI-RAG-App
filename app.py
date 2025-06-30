import streamlit as st

from src.openai_chain import OpenAIChain, OpenAIRAGChain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


@st.cache_resource
def load_chain(_chat_memory, pdf_chat=False, uploaded_file=None, db_type='pinecone'):
    """
    Load the appropriate chain based on whether PDF chat is enabled.
    The _chat_memory parameter is prefixed with underscore to tell Streamlit not to hash it.
    """
    try:
        if pdf_chat and uploaded_file:
            return OpenAIRAGChain(_chat_memory, uploaded_file=uploaded_file, db_type=db_type)
        else:
            return OpenAIChain(_chat_memory)
    except Exception as e:
        st.error(f"Failed to load chain: {str(e)}")
        return None


def file_uploader_change():
    """Handle file upload changes and update session state accordingly."""
    if st.session_state.uploaded_file:
        if not st.session_state.pdf_chat:
            clear_cache()
            st.session_state.pdf_chat = True
        st.session_state.knowledge_change = True
    else:
        clear_cache()
        st.session_state.pdf_chat = False


def toggle_pdf_chat_change():
    """Handle PDF chat toggle changes."""
    clear_cache()
    if st.session_state.pdf_chat and st.session_state.uploaded_file:
        st.session_state.knowledge_change = True


def database_change():
    """Handle database type changes."""
    clear_cache()
    if st.session_state.pdf_chat and st.session_state.uploaded_file:
        st.session_state.knowledge_change = True


def clear_input_field():
    """Store the question and clear the input field."""
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""


def set_send_input():
    """Set send input flag and clear input field."""
    st.session_state.send_input = True
    clear_input_field()


def clear_cache():
    """Clear all cached resources."""
    st.cache_resource.clear()


def initial_session_state():
    """Initialize session state variables."""
    st.session_state.send_input = False
    st.session_state.knowledge_change = False


def check_environment_variables():
    """Check if required environment variables are set."""
    import os
    missing_vars = []
    
    if not os.environ.get('OPENAI_API_KEY'):
        missing_vars.append('OPENAI_API_KEY')
    
    # Only check Pinecone if it's selected
    if st.session_state.get('database_type', 'pinecone') == 'pinecone':
        if not os.environ.get('PINECONE_API_KEY'):
            missing_vars.append('PINECONE_API_KEY')
    
    return missing_vars


def main():
    # Initialize
    st.title('Local Chat App')
    chat_container = st.container()

    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Database selection
    db_type = st.sidebar.selectbox(
        'Vector Database',
        ['pinecone', 'chroma'],
        key='database_type',
        on_change=database_change,
        help="Choose between Pinecone (cloud) and Chroma (local)"
    )
    
    # PDF chat toggle
    st.sidebar.toggle('PDF Chat', value=False, key='pdf_chat', on_change=toggle_pdf_chat_change)
    
    # File upload
    uploaded_pdf = st.sidebar.file_uploader('Upload your pdf files',
                                            type='pdf',
                                            accept_multiple_files=True,
                                            key='uploaded_file',
                                            on_change=file_uploader_change)

    # Check environment variables
    missing_vars = check_environment_variables()
    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        st.info("Please set these environment variables in your .env file or system environment.")
        if 'PINECONE_API_KEY' in missing_vars:
            st.info("💡 Tip: You can switch to Chroma (local database) in the sidebar to avoid needing Pinecone API key.")
        return

    # Input objects
    user_input = st.text_input('Type your message here', key='user_input', on_change=set_send_input)
    send_button = st.button('Send', key='send_button')

    # Initialize session state
    if 'send_input' not in st.session_state:
        initial_session_state()

    # Chat history
    chat_history = StreamlitChatMessageHistory(key='history')

    # Display chat history
    with chat_container:
        for msg in chat_history.messages:
            st.chat_message(msg.type).write(msg.content)

    # Load chain with proper error handling
    try:
        llm_chain = load_chain(
            _chat_memory=chat_history,
            pdf_chat=st.session_state.pdf_chat,
            uploaded_file=st.session_state.uploaded_file[0] if st.session_state.uploaded_file else None,
            db_type=db_type
        )
        
        if llm_chain is None:
            st.error("Failed to initialize chat chain. Please check your configuration.")
            st.info("If you're having issues with Pinecone, try switching to Chroma in the sidebar.")
            return
            
        # Update knowledge base if needed
        if st.session_state.knowledge_change and uploaded_pdf:
            with st.spinner('Updating knowledge base...'):
                try:
                    llm_chain.update_chain(uploaded_pdf[0])
                    st.session_state.knowledge_change = False
                    st.success("Knowledge base updated successfully!")
                except Exception as e:
                    st.error(f"Failed to update knowledge base: {str(e)}")
                    if db_type == 'pinecone':
                        st.info("This might be due to Pinecone configuration issues. Try switching to Chroma in the sidebar.")
                    st.session_state.knowledge_change = False

        # Handle user input
        if st.session_state.send_input and st.session_state.user_question:
            with chat_container:
                st.chat_message('user').write(st.session_state.user_question)
                try:
                    llm_response = llm_chain.run(user_input=st.session_state.user_question)
                    st.chat_message('ai').write(llm_response)
                except Exception as e:
                    st.error(f"Failed to get response: {str(e)}")
                finally:
                    st.session_state.user_question = ""
                    st.session_state.send_input = False
                    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("If you're experiencing issues, try restarting the application or check your API keys.")


if __name__ == '__main__':
    main()
