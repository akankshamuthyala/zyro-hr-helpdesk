import streamlit as st
import os
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DOCS_PATH = "docs/"

@st.cache_resource
def build_pipeline():
    documents = []
    for pdf_file in sorted(os.listdir(DOCS_PATH)):
        if not pdf_file.endswith(".pdf"):
            continue
        full_path = os.path.join(DOCS_PATH, pdf_file)
        reader = pypdf.PdfReader(full_path)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append(Document(
                    page_content=text.strip(),
                    metadata={"filename": pdf_file, "page": page_num}
                ))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 25, "lambda_mult": 0.5}
    )
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.1,
        max_tokens=1024,
        api_key=GROQ_API_KEY
    )
    prompt = ChatPromptTemplate.from_template(
        "You are an expert HR policy assistant for Zyro Dynamics Pvt. Ltd.\n"
        "IMPORTANT: Documents use both Zyro Dynamics and Acrux Dynamics — same company.\n"
        "Answer using ONLY the HR policy documents below.\n"
        "Include exact numbers, days, grades, policy codes.\n"
        "If not found say: I don't have specific information about that in our HR policy documents.\n\n"
        "HR Policy Documents:\n{context}\n\n"
        "Employee Question: {question}\n\nAnswer:"
    )
    def chain(question):
        docs = retriever.invoke(question)
        leave_keywords = ["casual leave", "earned leave", "sick leave", "how many leave", "leave entitlement", "days of leave"]
        if any(kw in question.lower() for kw in leave_keywords):
            extra_docs = vectorstore.similarity_search("ANNUAL LEAVE ENTITLEMENT casual leave 8 days", k=2)
            existing = [d.page_content for d in docs]
            for ed in extra_docs:
                if ed.page_content not in existing:
                    docs.insert(0, ed)
        context = "\n\n---\n\n".join(
            [f"[{d.metadata.get('filename','')}]\n{d.page_content}" for d in docs]
        )
        response = llm.invoke(prompt.invoke({"context": context, "question": question}))
        answer = StrOutputParser().invoke(response)
        sources = list({d.metadata.get("filename", "") for d in docs})
        return answer, sources
    return chain

HR_KEYWORDS = ["leave", "salary", "policy", "wfh", "work from home", "probation",
               "benefit", "conduct", "onboarding", "travel", "expense", "performance",
               "review", "compensation", "harassment", "separation", "notice",
               "resignation", "insurance", "pip", "appraisal", "increment", "ctc", "grade"]

REFUSAL = ("I'm sorry, I can only answer questions related to Zyro Dynamics HR policies. "
           "Please ask about leave, salary, WFH, performance, or other HR topics!")

def is_hr_question(q):
    return any(k in q.lower() for k in HR_KEYWORDS)

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢", layout="centered")
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about Zyro Dynamics HR policies")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.spinner("Loading HR documents..."):
    chain = build_pipeline()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if is_hr_question(user_input):
                response, sources = chain(user_input)
                if sources:
                    response += f"\n\n*Sources: {', '.join(set(sources))}*"
            else:
                response = REFUSAL
        st.markdown(response)
        st.session_state.messages.csv.append({"role": "assistant", "content": response})
