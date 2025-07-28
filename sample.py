# 🔐 API Key 환경변수에서 불러오기
import os
from dotenv import load_dotenv
load_dotenv()
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# 📦 주요 라이브러리 임포트
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 📁 PDF 루트 경로
PDF_ROOT = "C:/_knudata/hyundaicar_info"

# 📘 Streamlit 설정
st.set_page_config(page_title="현대차 RAG 데모", layout="wide")
st.title("🚗 현대자동차 GPT vs PDF RAG 비교 서비스")
st.caption("차량 카탈로그 및 가격표 PDF 기반 응답과 GPT 기본 응답 비교")

# 📂 차종 목록 불러오기 (하위 폴더 + PDF 포함 폴더 모두)
def get_all_models():
    model_list = []
    for category in os.listdir(PDF_ROOT):
        category_path = os.path.join(PDF_ROOT, category)
        if not os.path.isdir(category_path):
            continue

        has_pdf = any(f.endswith(".pdf") for f in os.listdir(category_path))
        if has_pdf:
            model_list.append((category, None))  # ex. ("승용", None)

        for model in os.listdir(category_path):
            model_path = os.path.join(category_path, model)
            if os.path.isdir(model_path):
                model_list.append((category, model))  # ex. ("승용", "그랜저")

    return model_list

# 🚘 모델 선택 UI
model_options = get_all_models()
selected_cat_model = st.selectbox("🚘 차종 선택", model_options, format_func=lambda x: f"{x[0]}" if x[1] is None else f"{x[0]} - {x[1]}")
selected_category, selected_model = selected_cat_model

# 📄 실제 PDF 폴더 경로 설정
if selected_model is None:
    model_dir = os.path.join(PDF_ROOT, selected_category)
    selected_model_name = selected_category
else:
    model_dir = os.path.join(PDF_ROOT, selected_category, selected_model)
    selected_model_name = selected_model

# 📄 PDF 경로 파싱
catalog_path, price_path = None, None
for file in os.listdir(model_dir):
    if "catalog" in file.lower():
        catalog_path = os.path.join(model_dir, file)
    elif "price" in file.lower():
        price_path = os.path.join(model_dir, file)

# 🧱 경로 점검
if not catalog_path or not price_path:
    st.error("카탈로그 또는 가격표 PDF 파일이 누락되었습니다.")
    st.stop()

st.write(f"📘 카탈로그 파일: `{os.path.basename(catalog_path)}`")
st.write(f"💰 가격표 파일: `{os.path.basename(price_path)}`")

# 📦 벡터스토어 캐시
@st.cache_resource
def load_vectorstores(catalog_path, price_path, car_id):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embedding = OpenAIEmbeddings()

    catalog_chunks = splitter.split_documents(PyPDFLoader(catalog_path).load())
    price_chunks = splitter.split_documents(PyPDFLoader(price_path).load())

    catalog_db = Chroma.from_documents(catalog_chunks, embedding=embedding, collection_name=f"{car_id}_catalog")
    price_db = Chroma.from_documents(price_chunks, embedding=embedding, collection_name=f"{car_id}_price")

    return catalog_db, price_db

catalog_db, price_db = load_vectorstores(catalog_path, price_path, selected_model_name)

# ✅ 질문 입력
question = st.text_input("❓ 차량에 대해 궁금한 점을 물어보세요", placeholder="예: 하이브리드 연비는 어떻게 되나요?")

# ✅ 응답 생성
if question:
    with st.spinner("🤖 응답 생성 중입니다..."):
        llm = ChatOpenAI(model="gpt-4o")

        # 1️⃣ GPT 기본 응답
        gpt_response = llm.invoke(question).content

        # 2️⃣ 카탈로그 기반 응답
        catalog_qa = RetrievalQA.from_chain_type(llm=llm, retriever=catalog_db.as_retriever())
        catalog_response = catalog_qa.invoke({"query": question})["result"]

        # 3️⃣ 가격표 기반 응답
        price_qa = RetrievalQA.from_chain_type(llm=llm, retriever=price_db.as_retriever())
        price_response = price_qa.invoke({"query": question})["result"]

    # 🧾 출력
    st.markdown("---")
    st.markdown(f"### 🚗 차량: `{selected_model_name}` | ❓ 질문: `{question}`")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🧠 **GPT 기본 응답**")
        st.write(gpt_response)
    with col2:
        st.markdown("📘 **카탈로그 기반 응답**")
        st.write(catalog_response)
    with col3:
        st.markdown("💰 **가격표 기반 응답**")
        st.write(price_response)
