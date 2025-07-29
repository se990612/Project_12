import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_anthropic import ChatAnthropic
import tempfile

# ✅ 환경 설정
load_dotenv()
st.set_page_config(page_title="📘 현대차 vs 타제조사 비교", layout="wide")
st.title("📘 Claude & GPT 기반 현대차 vs 타제조사 차량 비교 질의응답")

ROOT_DIR = "C:/_knudata/hyundaicar_info"
HYUNDAI_VECTOR_DIR = "C:/_knudata/vector_db/openai_catalog"

if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"):
    st.error("❌ .env에 ANTHROPIC_API_KEY 또는 OPENAI_API_KEY가 누락되어 있습니다.")
    st.stop()

# ✅ 현대차 목록 가져오기
def get_all_car_models():
    models = []
    for category in os.listdir(ROOT_DIR):
        category_path = os.path.join(ROOT_DIR, category)
        if os.path.isdir(category_path):
            for model in os.listdir(category_path):
                model_path = os.path.join(category_path, model)
                if os.path.isdir(model_path):
                    models.append((model_path, model))
    return models

# ✅ 현대차 벡터스토어 생성/로드
@st.cache_resource
def load_or_create_hyundai_vectorstore(catalog_path, price_path, save_dir):
    embeddings = OpenAIEmbeddings()
    index_file = os.path.join(save_dir, "index.faiss")
    pkl_file = os.path.join(save_dir, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(pkl_file):
        return FAISS.load_local(save_dir, embeddings, allow_dangerous_deserialization=True)

    docs = PyPDFLoader(catalog_path).load() + PyPDFLoader(price_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    for doc in chunks:
        doc.metadata['source'] = 'hyundai'

    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(save_dir)
    return vectordb

# ✅ 타제조사 PDF → 임시 Chroma 생성
def create_temp_chroma_vectorstore(uploaded_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        temp_path = tmp_file.name

    docs = PyPDFLoader(temp_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    for doc in chunks:
        doc.metadata['source'] = 'other'

    tmp_dir = tempfile.mkdtemp()
    return Chroma.from_documents(chunks, embedding=OpenAIEmbeddings(), persist_directory=tmp_dir)

# ✅ QA 체인
def build_qa_chain(model_name, retriever, provider="gpt"):
    if provider == "gpt":
        llm = ChatOpenAI(model_name=model_name, temperature=0)
    elif provider == "claude":
        llm = ChatAnthropic(model=model_name, temperature=0)
    else:
        raise ValueError("지원되지 않는 모델 제공자")
    return RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)

# ✅ 문서 추출
def get_top_chunks(query, retriever, k=5):
    return retriever.get_relevant_documents(query)[:k]

# ✅ UI 구성
car_model_map = get_all_car_models()
car_model_options = [model_name for _, model_name in car_model_map]
selected_model = st.selectbox("🚗 현대차 모델 선택", car_model_options)
uploaded_pdf = st.file_uploader("📄 타제조사 차량 PDF 업로드", type="pdf")
question = st.text_input("❓ 비교 질문 입력")

if selected_model and uploaded_pdf and question:
    rel_path, model_name = next((fp, dn) for fp, dn in car_model_map if dn == selected_model)
    catalog_path = os.path.join(rel_path, f"{model_name}-catalog.pdf")
    price_path = os.path.join(rel_path, f"{model_name}-price.pdf")

    if not os.path.exists(catalog_path) or not os.path.exists(price_path):
        st.error("❌ 현대차 PDF 누락")
    else:
        with st.spinner("📚 벡터 로딩 중"):
            vectordb_h = load_or_create_hyundai_vectorstore(catalog_path, price_path, os.path.join(HYUNDAI_VECTOR_DIR, model_name))
            vectordb_o = create_temp_chroma_vectorstore(uploaded_pdf)
            retriever_h = vectordb_h.as_retriever()
            retriever_o = vectordb_o.as_retriever()

        with st.spinner("🤖 Claude & GPT 응답 생성 중"):
            # Direct LLM
            gpt_direct = ChatOpenAI(model_name="gpt-4o", temperature=0)
            gpt_direct_answer = gpt_direct.invoke(question).content

            claude_direct = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
            claude_direct_answer = claude_direct.invoke(question).content

            # RAG
            gpt_chain_h = build_qa_chain("gpt-4o", retriever_h, "gpt")
            gpt_chain_o = build_qa_chain("gpt-4o", retriever_o, "gpt")
            gpt_h = gpt_chain_h.invoke({"question": question})
            gpt_o = gpt_chain_o.invoke({"question": question})

            claude_chain_h = build_qa_chain("claude-3-haiku-20240307", retriever_h, "claude")
            claude_chain_o = build_qa_chain("claude-3-haiku-20240307", retriever_o, "claude")
            claude_h = claude_chain_h.invoke({"question": question})
            claude_o = claude_chain_o.invoke({"question": question})

            top_chunks = get_top_chunks(question, retriever_h) + get_top_chunks(question, retriever_o)

        st.markdown(f"### ✅ 질문: {question}")
        st.subheader("🔍 기본 LLM 응답 (참조 없음)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("💬 GPT-4o 응답")
            st.info(gpt_direct_answer)
        with col2:
            st.markdown("💬 Claude 응답")
            st.info(claude_direct_answer)

        st.markdown("### 📄 RAG 기반 비교 응답")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("🧠 GPT - 현대차")
            st.success(gpt_h["answer"])
            st.caption(f"📚 출처: {gpt_h.get('sources', 'N/A')}")
            st.markdown("🧠 GPT - 타제조사")
            st.info(gpt_o["answer"])
        with col4:
            st.markdown("🧠 Claude - 현대차")
            st.success(claude_h["answer"])
            st.caption(f"📚 출처: {claude_h.get('sources', 'N/A')}")
            st.markdown("🧠 Claude - 타제조사")
            st.info(claude_o["answer"])

        with st.expander("📄 참조 문서 보기"):
            for i, doc in enumerate(top_chunks, 1):
                st.markdown(f"**문서 {i} ({doc.metadata.get('source', 'unknown')})**")
                st.code(doc.page_content.strip(), language="markdown")