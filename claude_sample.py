import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropicMessages

# ✅ 환경변수 로드 (.env에서 CLAUDE API KEY 읽기)
load_dotenv()
if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("❌ .env 파일에 ANTHROPIC_API_KEY가 설정되어 있지 않습니다.")
    st.stop()

# 📁 현대차 정보 폴더 (절대경로)
ROOT_DIR = "C:/_knudata/hyundaicar_info"

if not os.path.exists(ROOT_DIR):
    st.error(f"❌ 폴더를 찾을 수 없습니다: {ROOT_DIR}")
    st.stop()

# 🔍 전체 차종 탐색
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

# 🚘 차종 선택
car_model_map = get_all_car_models()
car_model_options = [model_name for _, model_name in car_model_map]
selected_models = st.multiselect("차종 선택 (1개 또는 2개)", car_model_options)

# 📄 PDF → VectorStore 변환 함수
@st.cache_resource
def load_vectorstore(catalog_path, price_path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for path in [catalog_path, price_path]:
        loader = PyPDFLoader(path)
        docs.extend(text_splitter.split_documents(loader.load()))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding=embeddings)
    return vectordb

# 🤖 Claude로 RAG 응답 생성
def answer_with_claude(vectorstore, query):
    retriever = vectorstore.as_retriever()
    llm = ChatAnthropicMessages(
        model="claude-3-5-sonnet-20240620",  # 정확한 이름
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    with st.spinner("Claude + RAG 응답 생성 중..."):
        return qa_chain.run(query)

# ✅ 단일 차량 선택
if len(selected_models) == 1:
    model_rel_path, model_name = next((fp, dn) for fp, dn in car_model_map if dn == selected_models[0])
    base_path = os.path.join(ROOT_DIR, model_rel_path)
    catalog_path = os.path.join(base_path, f"{model_name}-catalog.pdf")
    price_path = os.path.join(base_path, f"{model_name}-price.pdf")

    if not os.path.exists(catalog_path) or not os.path.exists(price_path):
        st.error("📂 해당 차종의 PDF 파일이 없습니다.")
    else:
        vectordb = load_vectorstore(catalog_path, price_path)
        st.subheader(f"📘 [{model_name.upper()}] 모델 질문해보세요")
        user_input = st.text_input("궁금한 내용을 입력하세요")

        if user_input:
            rag = answer_with_claude(vectordb, user_input)
            st.markdown("### 📌 Claude 기반 RAG 응답")
            st.success(rag)

# ✅ 두 개 선택 시 비교
elif len(selected_models) == 2:
    st.subheader(f"📊 차량 비교: {selected_models[0]} vs {selected_models[1]}")
    user_input = st.text_input("비교할 내용을 입력하세요 (예: 가격, 옵션, 연비 등)")

    def load_both_vectorstores():
        vectordbs = []
        for name in selected_models:
            rel_path, model_name = next((fp, dn) for fp, dn in car_model_map if dn == name)
            base_path = os.path.join(ROOT_DIR, rel_path)
            catalog_path = os.path.join(base_path, f"{model_name}-catalog.pdf")
            price_path = os.path.join(base_path, f"{model_name}-price.pdf")
            if os.path.exists(catalog_path) and os.path.exists(price_path):
                vectordbs.append(load_vectorstore(catalog_path, price_path))
        return vectordbs

    if user_input:
        vectordbs = load_both_vectorstores()
        if len(vectordbs) != 2:
            st.error("🚨 두 차량 모두 PDF가 존재해야 비교할 수 있어요.")
        else:
            rag1 = answer_with_claude(vectordbs[0], user_input)
            rag2 = answer_with_claude(vectordbs[1], user_input)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### 📘 {selected_models[0]} (Claude RAG)")
                st.success(rag1)
            with col2:
                st.markdown(f"### 📘 {selected_models[1]} (Claude RAG)")
                st.success(rag2)
