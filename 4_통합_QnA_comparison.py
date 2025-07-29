import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropicMessages

# ✅ 환경 설정
load_dotenv()
st.set_page_config(page_title="📘 Claude & GPT 차량 질의응답", layout="wide")
st.title("📘 Claude & GPT 기반 차량 카탈로그 + 가격표 RAG 통합 질문응답")

ROOT_DIR = "C:\Users\KDT13\kh0616\project_12\Project_12\hyundaicar_info"
VECTORSTORE_DIR = "C:\Users\KDT13\kh0616\project_12\Project_12\vector_db/combined"

if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"):
    st.error("❌ .env에 ANTHROPIC_API_KEY 또는 OPENAI_API_KEY가 누락되어 있습니다.")
    st.stop()

# ✅ 차량 목록 가져오기
def get_all_car_models():
    models = []
    for category in os.listdir(ROOT_DIR):
        category_path = os.path.join(ROOT_DIR, category)
        if os.path.isdir(category_path):
            for model in os.listdir(category_path):
                model_path = os.path.join(category_path, model)
                if os.path.isdir(model_path):
                    models.append((os.path.join(category, model), model))
    return models

# ✅ 벡터스토어 생성 또는 불러오기
@st.cache_resource
def load_or_create_combined_vectorstore(catalog_path, price_path, save_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_file = os.path.join(save_path, "index.faiss")
    pkl_file = os.path.join(save_path, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(pkl_file):
        # 🔧 보안 경고에 따른 명시적 허용
        return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(PyPDFLoader(catalog_path).load())
    chunks += splitter.split_documents(PyPDFLoader(price_path).load())

    vectordb = FAISS.from_documents(chunks, embeddings)
    os.makedirs(save_path, exist_ok=True)
    vectordb.save_local(save_path)
    return vectordb

# ✅ LLM 기반 QA 체인 생성
def build_qa_chain(model_name, retriever, provider="claude"):
    if provider == "claude":
        llm = ChatAnthropicMessages(model=model_name, temperature=0)
    elif provider == "openai":
        llm = ChatOpenAI(model_name=model_name, temperature=0)
    else:
        raise ValueError("지원되지 않는 모델 제공자")
    return RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)

# ✅ 관련 청크 보여주는 함수
def get_top_chunks(query, retriever, k=3):
    docs = retriever.get_relevant_documents(query)
    return docs[:k]

# ✅ UI 시작
car_model_map = get_all_car_models()
car_model_options = [model_name for _, model_name in car_model_map]
selected_model = st.selectbox("🚗 차량 선택", car_model_options)

if selected_model:
    rel_path, model_name = next((fp, dn) for fp, dn in car_model_map if dn == selected_model)
    catalog_path = os.path.join(ROOT_DIR, rel_path, f"{model_name}-catalog.pdf")
    price_path = os.path.join(ROOT_DIR, rel_path, f"{model_name}-price.pdf")

    if not os.path.exists(catalog_path):
        st.error("❌ 카탈로그 PDF가 존재하지 않습니다.")
    elif not os.path.exists(price_path):
        st.error("❌ 가격표 PDF가 존재하지 않습니다.")
    else:
        st.success(f"📄 `{model_name}`의 카탈로그 + 가격표 로드 완료")
        question = st.text_input("❓ 차량에 대해 궁금한 점을 입력하세요 (예: 가격, 옵션, 연비, 디자인 등)")

        if question:
            with st.spinner("🔍 Claude & GPT 벡터스토어 불러오는 중..."):
                vectordb_path = os.path.join(VECTORSTORE_DIR, model_name)
                vectordb = load_or_create_combined_vectorstore(catalog_path, price_path, vectordb_path)
                retriever = vectordb.as_retriever()

            with st.spinner("🤖 Claude & GPT 응답 생성 중..."):
                # Claude 응답
                qa_claude = build_qa_chain("claude-3-5-sonnet-20240620", retriever, provider="claude")
                result_claude = qa_claude(question)

                # GPT 응답
                qa_gpt = build_qa_chain("gpt-4", retriever, provider="openai")
                result_gpt = qa_gpt(question)

                # 근거 청크 추출 (상위 3개)
                top_chunks = get_top_chunks(question, retriever)

            st.markdown("## ✅ 질문")
            st.info(f"💬 {question}")

            st.markdown("## 🤖 Claude 응답")
