# 📄 pages/2_카탈로그_QnA.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropicMessages

# ✅ 환경 설정
load_dotenv()
st.set_page_config(page_title="📘 카탈로그 Q&A", layout="wide")
st.title("📘 차량 카탈로그 기반 Claude RAG 질문응답")

ROOT_DIR = "C:/_knudata/hyundaicar_info"
VECTORSTORE_DIR = "C:/_knudata/vector_db/catalog"

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("❌ .env에 ANTHROPIC_API_KEY 설정 필요")
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

car_model_map = get_all_car_models()
car_model_options = [model_name for _, model_name in car_model_map]
selected_model = st.selectbox("🚗 차량 선택", car_model_options)

# ✅ FAISS 저장/불러오기
@st.cache_resource
def load_or_create_faiss(pdf_path, save_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    index_file = os.path.join(save_path, "index.faiss")
    pkl_file = os.path.join(save_path, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(pkl_file):
        return FAISS.load_local(save_path, embeddings)
    else:
        docs = PyPDFLoader(pdf_path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        vectordb = FAISS.from_documents(chunks, embeddings)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        vectordb.save_local(save_path)
        return vectordb

# ✅ Claude RAG 응답 함수
def answer_with_claude(vectordb, query):
    retriever = vectordb.as_retriever()
    llm = ChatAnthropicMessages(model="claude-3-5-sonnet-20240620", temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)

# ✅ PDF 경로 및 벡터 경로
if selected_model:
    rel_path, model_name = next((fp, dn) for fp, dn in car_model_map if dn == selected_model)
    catalog_path = os.path.join(ROOT_DIR, rel_path, f"{model_name}-catalog.pdf")

    if not os.path.exists(catalog_path):
        st.error("❌ 해당 차량의 카탈로그 PDF가 존재하지 않습니다.")
    else:
        st.success(f"📄 {model_name} 카탈로그 불러오기 완료")
        question = st.text_input("❓ 카탈로그 기반 질문을 입력하세요")

        if question:
            with st.spinner("Claude RAG 응답 생성 중..."):
                faiss_dir = os.path.join(VECTORSTORE_DIR, model_name)
                vectordb = load_or_create_faiss(catalog_path, faiss_dir)
                response = answer_with_claude(vectordb, question)

            st.markdown("---")
            st.markdown(f"### 🚗 차량: `{model_name}`")
            st.markdown(f"### ❓ 질문: `{question}`")
            st.markdown("---")
            st.markdown("### 🤖 Claude RAG 응답")
            st.write(response)
