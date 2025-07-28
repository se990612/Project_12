# 안전한 예시 (.env에서 불러오기)
import os
from dotenv import load_dotenv
load_dotenv()

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 📄 PDF 경로
catalog_path = "grandeur-catalog.pdf"
price_path = "grandeur-2026-price.pdf"

# 📘 Streamlit 설정
st.set_page_config(page_title="RAG 비교 데모", layout="wide")
st.title("📊 GPT-4o 단독 vs RAG PDF 기반 응답 비교")
st.caption("카탈로그/가격표 PDF를 활용한 질의응답 성능 비교")

# 📦 벡터스토어 로딩
@st.cache_resource
def load_vectorstores():
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embedding = OpenAIEmbeddings()

    catalog_chunks = splitter.split_documents(PyPDFLoader(catalog_path).load())
    price_chunks = splitter.split_documents(PyPDFLoader(price_path).load())

    catalog_db = Chroma.from_documents(catalog_chunks, embedding=embedding, collection_name="grandeur_catalog")
    price_db = Chroma.from_documents(price_chunks, embedding=embedding, collection_name="grandeur_price")

    return catalog_db, price_db

catalog_db, price_db = load_vectorstores()

# ✅ 사용자 질문 입력
question = st.text_input("❓ 비교할 질문을 입력하세요", placeholder="예: 하이브리드 모델의 출력은 어떻게 되나요?")

# ✅ 응답 생성 및 출력
if question:
    with st.spinner("🤖 GPT-4o가 응답을 생성 중입니다..."):
        llm = ChatOpenAI(model_name="gpt-4o")
        
        # 1️⃣ GPT 기본 응답
        gpt_response = llm.invoke(question).content

        # 2️⃣ 카탈로그 RAG 응답
        catalog_qa = RetrievalQA.from_chain_type(llm=llm, retriever=catalog_db.as_retriever())
        catalog_response = catalog_qa.invoke({"query": question})["result"]

        # 3️⃣ 가격표 RAG 응답
        price_qa = RetrievalQA.from_chain_type(llm=llm, retriever=price_db.as_retriever())
        price_response = price_qa.invoke({"query": question})["result"]

    st.markdown("---")
    st.markdown(f"### ❓ 질문: {question}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🧠 **GPT 기본 응답**")
        st.write(gpt_response)

    with col2:
        st.markdown("📘 **카탈로그 PDF 기반 응답**")
        st.write(catalog_response)

    with col3:
        st.markdown("💰 **가격표 PDF 기반 응답**")
        st.write(price_response)
