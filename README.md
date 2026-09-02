# College Document Assistant

AI-powered College Document Assistant that allows users to upload college PDF documents and ask questions using semantic document search.

## Features

- Upload PDF documents
- Extract text from PDFs
- Split documents into smaller chunks
- Generate embeddings using HuggingFace
- Store embeddings using FAISS
- Perform semantic similarity search
- Retrieve relevant document sections
- Display source page numbers
- Professional Streamlit interface
- Temporary PDF processing without hardcoded file paths

## Technologies Used

- Python
- Streamlit
- LangChain
- HuggingFace Embeddings
- FAISS
- PyPDF
- Sentence Transformers

## RAG Workflow

PDF Upload → Text Extraction → Text Splitting → Embeddings → FAISS Vector Database → Similarity Search → Relevant Content

## Project Structure

College-Document-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

## Installation

Clone the repository:

    git clone https://github.com/Udhayaprabhas2904/college-document-assistant.git

Navigate to the project:

    cd college-document-assistant

Create a virtual environment:

    python -m venv venv

Activate the virtual environment on Windows:

    venv\Scripts\activate

Install the required packages:

    pip install -r requirements.txt

## Run the Application

Start the Streamlit application:

    python -m streamlit run app.py

The application will open in your browser.

## How It Works

1. Upload a college PDF document.
2. The PDF text is extracted using PyPDFLoader.
3. The extracted text is divided into smaller chunks.
4. HuggingFace converts the chunks into embeddings.
5. FAISS stores the embeddings for fast similarity search.
6. The user's question is converted into an embedding.
7. FAISS retrieves the most relevant document sections.
8. The retrieved content is displayed with the corresponding page numbers.

## Example Questions

- What are the college attendance rules?
- What is the minimum attendance requirement?
- What are the examination regulations?
- What are the internal assessment rules?
- What are the admission requirements?

## Current Implementation

The current version focuses on document retrieval. It retrieves relevant content from the uploaded PDF using semantic similarity search.

## Future Enhancements

- Integrate an LLM for natural-language answers
- Add conversational memory
- Support multiple PDF documents
- Add citation-based responses
- Add OCR for scanned PDFs
- Add similarity scores
- Add document management
- Deploy the application online

## License

This project is developed for educational and learning purposes.
