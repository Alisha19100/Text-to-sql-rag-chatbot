# Text-to-SQL-RAG-Chatbot

RAG-based Text-to-SQL system for natural-language database querying with LangChain, FAISS, Groq, MySQL, and CSV-to-SQLite support.

The system allows users to query databases using natural language instead of writing SQL manually. RAG is used to retrieve relevant database schema and business context before generating SQL, helping the LLM understand the available tables, columns, and likely relationships.

## Features

- Natural language to SQL generation
- RAG-based schema and metadata retrieval
- FAISS vector search
- LangChain-based workflow
- Groq LLM integration
- Read-only SQL execution
- SQL validation and correction
- Conversation-aware follow-up questions
- MySQL database support
- CSV upload for custom datasets
- Automatic CSV-to-SQLite database creation
- Support for multiple CSV files
- Automatic detection of likely relationships between uploaded tables
- Streamlit-based interactive interface

## How It Works

```text
User Question
      ↓
RAG Retriever
      ↓
Relevant Schema + Business Context
      ↓
Groq LLM
      ↓
SQL Generation
      ↓
SQL Validation
      ↓
Database Execution
      ↓
Query Result
      ↓
Natural Language Answer
Natural Language Answer
