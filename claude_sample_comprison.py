import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropicMessages

# 🔐 API KEY 로드
load_dotenv()
if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("❌ .env 파일에 ANTHROPIC_API_KEY가 설정되어 있지 않습니다.")
    st.stop()

# 📁 현대차 PDF 경로
ROOT_DIR = "C:/_knudata/hyundaicar_info"
if not os.path.exists(ROOT_DIR):
    st.error(f"❌ 폴더를 찾을 수 없습니다: {ROOT_DIR}")
    st.stop()

# 🔍 전체 차량 리스트
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
selected_models = st.multiselect("🚘 차종 선택 (1개 또는 2개)", car_model_options)

# 🔄 PDF → 하나의 벡터스토어 생성
@st.cache_resource
def load_vectorstore_combined(pdf_paths):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_docs = []
    for path in pdf_paths:
        docs = PyPDFLoader(path).load()
        chunks = splitter.split_documents(docs)
        all_docs.extend(chunks)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(all_docs, embedding=embeddings)
    return vectordb

# 🤖 Claude 응답
def answer_with_claude(vectorstore, query):
    retriever = vectorstore.as_retriever()
    llm = ChatAnthropicMessages(
        model="claude-3-5-sonnet-20240620",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(query)

# ✅ 차량 1개: 기본 응답 + Hyundai RAG 응답
if len(selected_models) == 1:
    model_rel_path, model_name = next((fp, dn) for fp, dn in car_model_map if dn == selected_models[0])
    base_path = os.path.join(ROOT_DIR, model_rel_path)
    catalog_path = os.path.join(base_path, f"{model_name}-catalog.pdf")
    price_path = os.path.join(base_path, f"{model_name}-price.pdf")

    if not os.path.exists(catalog_path) or not os.path.exists(price_path):
        st.error("❌ 카탈로그 또는 가격표 PDF가 없습니다.")
    else:
        vectordb = load_vectorstore_combined([catalog_path, price_path])

        st.subheader(f"📘 [{model_name}] 차량 질문")
        question = st.text_input("❓ 궁금한 점을 입력하세요")

        if question:
            with st.spinner("Claude 응답 생성 중..."):
                llm = ChatAnthropicMessages(
                    model="claude-3-5-sonnet-20240620",
                    temperature=0,
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )

                # 1️⃣ Claude 기본 응답
                base_response = llm.invoke(question).content

                # 2️⃣ Hyundai RAG
                rag_qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
                rag_response = rag_qa.invoke({"query": question})["result"]

            # 🔍 응답 비교 출력
            st.markdown("---")
            st.markdown(f"### 🚗 차량: `{model_name}` | ❓ 질문: `{question}`")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("🧠 **Claude 기본 응답**")
                st.write(base_response)
            with col2:
                st.markdown("🔍 **현대차 PDF 기반 RAG 응답**")
                st.write(rag_response)

# ✅ 차량 2개 선택: Hyundai RAG 응답 비교
elif len(selected_models) == 2:
    st.subheader(f"📊 차량 비교: {selected_models[0]} vs {selected_models[1]}")
    question = st.text_input("비교할 질문을 입력하세요")

    def load_both_vectorstores():
        vectordbs = []
        for name in selected_models:
            rel_path, model_name = next((fp, dn) for fp, dn in car_model_map if dn == name)
            base_path = os.path.join(ROOT_DIR, rel_path)
            catalog_path = os.path.join(base_path, f"{model_name}-catalog.pdf")
            price_path = os.path.join(base_path, f"{model_name}-price.pdf")
            if os.path.exists(catalog_path) and os.path.exists(price_path):
                vectordb = load_vectorstore_combined([catalog_path, price_path])
                vectordbs.append((model_name, vectordb))
        return vectordbs

    if question:
        db_pairs = load_both_vectorstores()
        if len(db_pairs) != 2:
            st.error("❗ 두 차량의 카탈로그/가격표 PDF가 모두 존재해야 합니다.")
        else:
            with st.spinner("Claude RAG 비교 응답 생성 중..."):
                rag1 = answer_with_claude(db_pairs[0][1], question)
                rag2 = answer_with_claude(db_pairs[1][1], question)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### 📘 {db_pairs[0][0]} (현대차 RAG 응답)")
                st.write(rag1)
            with col2:
                st.markdown(f"### 📘 {db_pairs[1][0]} (현대차 RAG 응답)")
                st.write(rag2)
