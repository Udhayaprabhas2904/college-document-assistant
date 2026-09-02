import os
import hashlib
import tempfile

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



# PAGE CONFIGURATION


st.set_page_config(
    page_title="College Document Assistant",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded"
)



# PROFESSIONAL THEME


st.markdown(
    """
    <style>

    /* GLOBAL */

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
    );

    html, body, [class*="css"] {
        font-family: "Inter", -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif;
    }

    .stApp {
        background: #F5F5F3;
        color: #18181B;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 42px;
        padding-bottom: 50px;
    }


    /* HEADINGS */

    h1 {
        color: #111111 !important;
        font-size: 34px !important;
        font-weight: 700 !important;
        letter-spacing: -1px !important;
        margin-bottom: 8px !important;
    }

    h2 {
        color: #18181B !important;
        font-size: 24px !important;
        font-weight: 650 !important;
        letter-spacing: -0.4px !important;
    }

    h3 {
        color: #18181B !important;
        font-size: 18px !important;
        font-weight: 600 !important;
    }

    p {
        color: #52525B;
        line-height: 1.65;
    }


    /* SIDEBAR */

    section[data-testid="stSidebar"] {
        background: #181818;
        border-right: 1px solid #292929;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 28px;
    }

    section[data-testid="stSidebar"] * {
        color: #F5F5F5 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #383838 !important;
        margin: 22px 0;
    }

    section[data-testid="stSidebar"] .stCaption {
        color: #A1A1AA !important;
    }

    section[data-testid="stSidebar"] .stAlert {
        background: #242424;
        border: 1px solid #383838;
    }


    /* SIDEBAR BRAND */

    .sidebar-brand {
        padding: 4px 0 8px 0;
    }

    .sidebar-brand-title {
        font-size: 19px;
        font-weight: 700;
        letter-spacing: -0.3px;
    }

    .sidebar-brand-subtitle {
        font-size: 12px;
        color: #A1A1AA;
        margin-top: 4px;
    }


    /* WELCOME AREA */

    .welcome-label {
        display: inline-block;
        background: #E7E7E4;
        color: #3F3F46;
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    .welcome-title {
        color: #111111;
        font-size: 38px;
        line-height: 1.15;
        font-weight: 700;
        letter-spacing: -1.4px;
        margin-bottom: 10px;
    }

    .welcome-description {
        max-width: 720px;
        color: #626262;
        font-size: 15px;
        line-height: 1.7;
    }


    /* SECTION LABEL */

    .section-label {
        color: #18181B;
        font-size: 13px;
        font-weight: 650;
        letter-spacing: 0.2px;
        margin-bottom: 8px;
    }


    /* CARDS */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border: 1px solid #E4E4E1;
        border-radius: 12px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }


    /* UPLOADER */

   [data-testid="stFileUploader"] {
    background: #FAFAF8;
    border: 1px dashed #B8B8B3;
    border-radius: 10px;
    padding: 8px;
}

[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF;
    border-radius: 7px;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #181818 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploaderDropzone"] button *,
[data-testid="stFileUploaderDropzone"] button span {
    color: #FFFFFF !important;
}

    /* TEXT INPUT */

    div[data-baseweb="input"] > div {
        background: #FFFFFF;
        border: 1px solid #D4D4D0;
        border-radius: 8px;
    }

    div[data-baseweb="input"] > div:focus-within {
        border: 1px solid #181818;
        box-shadow: 0 0 0 1px #181818;
    }

    input {
        color: #18181B !important;
    }

    input::placeholder {
        color: #A1A1AA !important;
    }


    /* BUTTONS */

   /* Normal Streamlit buttons */
.stButton > button {
    background: #181818 !important;
    color: #FFFFFF !important;
    border: 1px solid #181818 !important;
    border-radius: 8px !important;
    min-height: 42px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}

/* Normal button hover */
.stButton > button:hover {
    background: #333333 !important;
    border-color: #333333 !important;
}


/* SEND BUTTON */

button[kind="primaryFormSubmit"] {
    background: #16A34A !important;
    color: #FFFFFF !important;
    border: 1px solid #16A34A !important;
    border-radius: 8px !important;
    min-height: 42px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}

/* Send button hover */
button[kind="primaryFormSubmit"]:hover {
    background: #15803D !important;
    border-color: #15803D !important;
    color: #FFFFFF !important;
}

/* Send icon */
button[kind="primaryFormSubmit"] svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
}


    /* METRICS */

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E4E4E1;
        border-radius: 10px;
        padding: 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #71717A !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #18181B !important;
        font-size: 25px !important;
        font-weight: 650 !important;
    }


    /* CHAT */

    [data-testid="stChatMessage"] {
        background: #FFFFFF;
        border: 1px solid #E4E4E1;
        border-radius: 10px;
        margin-bottom: 10px;
    }


    /* EXPANDERS */

    div[data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #E4E4E1;
        border-radius: 9px;
    }

    div[data-testid="stExpander"] summary {
        font-size: 13px;
        font-weight: 600;
    }


    /* ALERTS */

    div[data-testid="stAlert"] {
        border-radius: 8px;
    }


    /* DIVIDER */

    hr {
        border-color: #E4E4E1 !important;
    }


    /* FOOTER */

    .footer {
        text-align: center;
        color: #A1A1AA;
        font-size: 11px;
        padding-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# SESSION STATE


if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "documents" not in st.session_state:
    st.session_state.documents = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []



# SIDEBAR


with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">
                College Assistant
            </div>
            <div class="sidebar-brand-subtitle">
                Document Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### RAG Pipeline")

    st.markdown(
        """
        **01  Upload**  
        Add your college PDF document.

        **02  Extract**  
        Extract readable text from the PDF.

        **03  Split**  
        Divide the content into smaller chunks.

        **04  Embed**  
        Convert text into vector representations.

        **05  Retrieve**  
        Find the most relevant content.
        """
    )

    st.divider()

    if st.session_state.vector_db is not None:

        st.success("Document ready")

        st.caption(
            f"File: {st.session_state.file_name}"
        )

    else:

        st.info(
            "No document loaded."
        )

    st.divider()

    st.caption(
        "LangChain  •  FAISS  •  HuggingFace"
    )



# MAIN WELCOME SECTION


st.markdown(
    '<div class="welcome-label">DOCUMENT INTELLIGENCE</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="welcome-title">'
    'College Document Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="welcome-description">'
    'Upload a college document and ask questions using '
    'semantic document search. The assistant retrieves '
    'relevant information directly from your uploaded PDF.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")
st.write("")



# UPLOAD DOCUMENT


with st.container(border=True):

    st.markdown(
        '<div class="section-label">'
        'DOCUMENT'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader("Upload your PDF")

    st.caption(
        "Choose a PDF from your computer. "
        "The document will be processed automatically."
    )

    uploaded_file = st.file_uploader(
        "Choose PDF document",
        type=["pdf"],
        label_visibility="collapsed"
    )



# PROCESS PDF


if uploaded_file is not None:

    
    # CREATE HASH FOR UPLOADED FILE
   

    file_bytes = uploaded_file.getvalue()

    current_file_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()


    
    # PROCESS ONLY NEW FILE
    

    if (
        st.session_state.file_hash
        != current_file_hash
    ):

        pdf_path = None

        try:

            
            # SAVE UPLOAD TO TEMPORARY FILE
         

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(file_bytes)
                pdf_path = temp_file.name


         
            # LOAD PDF
            

            with st.spinner(
                "Reading document..."
            ):

                loader = PyPDFLoader(
                    pdf_path
                )

                documents = loader.load()


            if not documents:

                st.error(
                    "The uploaded PDF does not contain readable content."
                )

                st.stop()


            
            # SPLIT DOCUMENT
           

            with st.spinner(
                "Preparing document..."
            ):

                text_splitter = (
                    RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )
                )

                chunks = (
                    text_splitter
                    .split_documents(documents)
                )


            if not chunks:

                st.error(
                    "No readable text was found in the PDF."
                )

                st.stop()


            # CREATE EMBEDDINGS
           

            with st.spinner(
                "Creating document embeddings..."
            ):

                embeddings = HuggingFaceEmbeddings(
                    model_name=(
                        "sentence-transformers/"
                        "all-MiniLM-L6-v2"
                    )
                )


     
            # CREATE FAISS DATABASE
          

            with st.spinner(
                "Building searchable index..."
            ):

                vector_db = FAISS.from_documents(
                    chunks,
                    embeddings
                )


           
            # SAVE SESSION DATA
           

            st.session_state.vector_db = vector_db
            st.session_state.documents = documents
            st.session_state.chunks = chunks
            st.session_state.file_name = (
                uploaded_file.name
            )
            st.session_state.file_hash = (
                current_file_hash
            )

            # Clear old conversation
            st.session_state.messages = []
            st.session_state.last_sources = []


            st.success(
                "Document processed successfully."
            )


        except Exception as e:

            st.error(
                f"Unable to process the PDF: {e}"
            )


        finally:

            if (
                pdf_path is not None
                and os.path.exists(pdf_path)
            ):

                os.remove(pdf_path)



# DOCUMENT INFORMATION


if st.session_state.vector_db is not None:

    st.write("")
    st.write("")

    st.markdown(
        '<div class="section-label">'
        'DOCUMENT OVERVIEW'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            label="Status",
            value="Ready"
        )

    with col2:

        st.metric(
            label="Pages",
            value=len(
                st.session_state.documents
            )
        )

    with col3:

        st.metric(
            label="Text Chunks",
            value=len(
                st.session_state.chunks
            )
        )


    st.write("")
    st.write("")


    
    # ACTIVE DOCUMENT
    

    with st.container(border=True):

        st.markdown(
            '<div class="section-label">'
            'ACTIVE DOCUMENT'
            '</div>',
            unsafe_allow_html=True
        )

        st.subheader(
            st.session_state.file_name
        )

        st.caption(
            "This document is currently available "
            "for semantic search."
        )


    st.write("")
    st.write("")


    
    # CHAT SECTION
   

    st.markdown(
        '<div class="section-label">'
        'ASSISTANT'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader("Ask about your document")

    st.caption(
        "Ask a question based on the information "
        "contained in the uploaded PDF."
    )

    st.write("")


   
    # PREVIOUS MESSAGES
   

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


    
    # QUESTION FORM
    

    with st.form(
        key="question_form",
        clear_on_submit=True
    ):

        question = st.text_input(
            "Question",
            placeholder=(
                "e.g. What are the college attendance rules?"
            ),
            label_visibility="collapsed"
        )

        send_button = st.form_submit_button(
            "Send",
            use_container_width=True
        )


    
    # HANDLE QUESTION
    

    if send_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            question = question.strip()


            
            # USER MESSAGE
            

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )


           
            # SEARCH VECTOR DATABASE
            

            with st.spinner(
                "Searching document..."
            ):

                retrieved_docs = (
                    st.session_state
                    .vector_db
                    .similarity_search(
                        question,
                        k=4
                    )
                )


           
            # STORE SOURCES
           

            st.session_state.last_sources = (
                retrieved_docs
            )


           
            # CREATE RESPONSE
           

            answer_parts = []

            for doc in retrieved_docs:

                if doc.page_content.strip():

                    answer_parts.append(
                        doc.page_content.strip()
                    )


            if answer_parts:

                answer = "\n\n".join(
                    answer_parts
                )

            else:

                answer = (
                    "No relevant information was found "
                    "in the uploaded document."
                )


          
            # ASSISTANT MESSAGE
           
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )


            st.rerun()


    
    # SOURCES
    

    if st.session_state.last_sources:

        st.write("")
        st.write("")

        st.markdown(
            '<div class="section-label">'
            'SOURCES'
            '</div>',
            unsafe_allow_html=True
        )

        st.subheader(
            "Retrieved document sections"
        )

        st.caption(
            "The following sections were identified "
            "as relevant to your question."
        )


        for i, doc in enumerate(
            st.session_state.last_sources
        ):

            page_number = (
                doc.metadata.get(
                    "page",
                    0
                ) + 1
            )

            with st.expander(
                f"Source {i + 1}    •    Page {page_number}"
            ):

                st.write(
                    doc.page_content
                )

                st.caption(
                    f"Document page {page_number}"
                )



# WELCOME STATE


else:

    st.write("")
    st.write("")

    with st.container(border=True):

        st.markdown(
            '<div class="section-label">'
            'GET STARTED'
            '</div>',
            unsafe_allow_html=True
        )

        st.subheader(
            "Upload a document to begin"
        )

        st.write(
            "Your PDF will be converted into searchable "
            "text and indexed using vector embeddings."
        )

        st.caption(
            "Supported format: PDF"
        )



# FOOTER


st.write("")
st.write("")

st.divider()

st.markdown(
    """
    <div class="footer">
        College Document Assistant
        &nbsp;•&nbsp;
        Semantic Document Search
        &nbsp;•&nbsp;
        LangChain + FAISS + HuggingFace
    </div>
    """,
    unsafe_allow_html=True
)