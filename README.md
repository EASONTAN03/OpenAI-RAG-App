# Multimodal RAG Chat App

A sophisticated Retrieval-Augmented Generation (RAG) chatbot application that supports multimodal interactions with PDF documents. Built with Streamlit, LangChain, and OpenAI, this application provides intelligent document analysis and conversational AI capabilities.

## 🌟 Features

- **📄 PDF Document Processing**: Upload and analyze PDF documents with intelligent text extraction
- **🤖 AI-Powered Conversations**: Engage in natural conversations with context-aware responses
- **🔍 Advanced RAG System**: History-aware retrieval with question-answering chains
- **💾 Dual Database Support**: Choose between Pinecone (cloud) and Chroma (local) vector databases
- **🔄 Real-time Knowledge Updates**: Dynamic knowledge base updates when new documents are uploaded
- **🎯 Context-Aware Responses**: Maintains conversation context for more relevant answers
- **⚡ Streamlit Interface**: Modern, responsive web interface for seamless user experience

## 📋 Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Configuration](#configuration)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)


## 🏗️ Architecture

### Project Structure
```
Multimodal-RAG-Chat-App/
├── src/
│   ├── openai_chain.py      # LLM chain implementations
│   ├── vectorstore.py       # Vector database management
│   ├── pdf_handler.py       # PDF processing utilities
│   └── utils.py            # Configuration and utility functions
├── assets/                 # Application screenshots and images
├── app.py                  # Main Streamlit application
├── config.yaml            # Application configuration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.12+** (recommended)
- **Conda** or **Miniconda** for environment management
- **Git** for cloning the repository
- **OpenAI API Key** for AI model access
- **Pinecone API Key** (optional, for cloud vector database)

## 🚀 Installation

## Getting Started

To get started with Multimodal RAG Chatbot, follow these simple steps:

**1. Clone this project**

```bash
git clone https://github.com/EASONTAN03/OpenAI-RAG-App.git
cd Multimodal-RAG-Chat-App
```

### 2. Create and Activate Conda Environment

```bash
# Create a new conda environment with Python 3.12
conda create -n multimodal-rag-chat python=3.12

# Activate the environment
conda activate multimodal-rag-chat
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root directory:

```bash
# Create .env file
touch .env
```

Add the following environment variables to your `.env` file:

```env
# Required: OpenAI API Key for chat and embeddings
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Pinecone API Key (only needed if using Pinecone database)
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 5. Configure the Application

Edit `config.yaml` to customize your settings:

```yaml
# OpenAI API Configuration
chat_model:
  model_name: "gpt-4o"          # or "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 1024

# Embedding Model Configuration
embedding_model:
  model_name: "text-embedding-3-large"
  dimensions: 1536

vector_database:
  chroma:

chat_session_path: './chat_session/'
```

## 🔑 Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `OPENAI_API_KEY` | ✅ Yes | Your OpenAI API key for chat and embedding models | `sk-...` |
| `PINECONE_API_KEY` | ❌ No* | Your Pinecone API key for cloud vector database | `...` |

*Pinecone API key is only required if you choose to use Pinecone as your vector database. You can use Chroma (local database) without this key.

### Getting API Keys

#### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to "API Keys" section
4. Create a new API key
5. Copy the key and add it to your `.env` file

#### Pinecone API Key (Optional)
1. Visit [Pinecone Console](https://app.pinecone.io/)
2. Sign up or log in to your account
3. Navigate to "API Keys" section
4. Create a new API key
5. Copy the key and add it to your `.env` file

## 🎯 Usage

### Starting the Application

```bash
# Make sure you're in the project directory and conda environment is activated
conda activate multimodal-rag-chat

# Start the Streamlit application
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Application

#### 1. Database Selection
- **Pinecone**: Cloud-based vector database (requires API key)
- **Chroma**: Local vector database (no API key required)

#### 2. Chat Modes

**PDF Chat Mode** (Recommended for document analysis):
- Toggle "PDF Chat" in the sidebar
- Upload one or more PDF files
- Ask questions about the uploaded documents
- The AI will reference the document content for accurate answers

![pdf chat](./assets/pdf-chat-mode.png)

**Regular Chat Mode**:
- Keep "PDF Chat" toggled off
- Ask general questions
- The AI will use its built-in knowledge

![non pdf chat](./assets/non-pdf-chat-mode.png)

#### 3. File Upload
- Supported format: PDF files
- Multiple files can be uploaded
- Files are automatically processed and indexed

#### 4. Asking Questions
- Type your question in the text input field
- Press Enter or click "Send"
- The AI will provide contextually relevant answers


### RAG System Components

#### 1. Indexing Stage
- **Document Processing**: PDF text extraction and chunking
- **Embedding Generation**: Text chunks converted to vector embeddings
- **Vector Storage**: Embeddings stored in vector database (Pinecone/Chroma)

![Indexing phase](./assets/indexing.png)

#### 2. Querying Stage
- **History-Aware Retrieval**: Contextual query enhancement using chat history
- **Document Retrieval**: Relevant documents fetched from vector database
- **Answer Generation**: LLM generates final answer using retrieved context

![Querying stage](./assets/querying.png)

### Key Components

- **`app.py`**: Main Streamlit application with UI and state management
- **`src/openai_chain.py`**: LLM chain implementations for regular and RAG chat
- **`src/vectorstore.py`**: Vector database management (Pinecone/Chroma)
- **`src/pdf_handler.py`**: PDF processing and text extraction
- **`src/utils.py`**: Utility functions and configuration loading

## 🔧 Configuration

### Model Configuration
You can customize the AI models in `config.yaml`:

```yaml
chat_model:
  model_name: "gpt-4o"          # Chat model
  temperature: 0.7              # Creativity level (0.0-1.0)
  max_tokens: 1024              # Maximum response length

embedding_model:
  model_name: "text-embedding-3-large"  # Embedding model
  dimensions: 1536              # Vector dimensions
```

### Database Configuration
- **Pinecone**: Cloud-based, requires API key, better for production
- **Chroma**: Local storage, no API key required, good for development

## 🐛 Troubleshooting

### Common Issues

#### 1. Missing Environment Variables
**Problem**: Application shows error about missing API keys
**Solution**: 
- Ensure `.env` file exists in project root
- Verify API keys are correctly set
- Restart the application after adding environment variables

#### 2. Pinecone Configuration Issues
**Problem**: Errors related to Pinecone API or dimensions
**Solution**:
- Switch to Chroma database in the sidebar
- Verify Pinecone API key is correct
- Check Pinecone index configuration

#### 3. PDF Processing Errors
**Problem**: Issues with PDF file upload or processing
**Solution**:
- Ensure PDF files are not corrupted
- Check file size (recommended < 50MB)
- Verify PDF contains extractable text

#### 4. Memory Issues
**Problem**: Application becomes slow or crashes
**Solution**:
- Clear browser cache
- Restart the Streamlit application
- Use Chroma instead of Pinecone for large documents

### Performance Tips

- Use Chroma for development and testing
- Switch to Pinecone for production deployments
- Clear cache if you encounter issues with chain reloading
- Restart the application if you change database types
- Limit PDF file sizes for better performance

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/OpenAI-RAG-App.git
cd Multimodal-RAG-Chat-App

# Create development environment
conda create -n multimodal-rag-dev python=3.12
conda activate multimodal-rag-dev

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Set up pre-commit hooks (optional)
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions, issues, or contributions:

- **Email**: Buitanphuong712@gmail.com (Original Author)
- **GitHub Issues**: [Create an issue](https://github.com/EASONTAN03/OpenAI-RAG-App/issues)
- **Original Repository**: [PhuongBui712/Multimodal-RAG-Chat-App](https://github.com/PhuongBui712/Multimodal-RAG-Chat-App)

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [LangChain](https://langchain.com/) for the RAG implementation
- [OpenAI](https://openai.com/) for the AI models
- [Pinecone](https://pinecone.io/) for vector database services
- [Chroma](https://www.trychroma.com/) for local vector storage

---

**Note**: This application requires an active internet connection for OpenAI API calls and optional Pinecone database access.