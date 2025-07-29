# 📄 pages/5_타제조사_정보비교.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropicMessages
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile

# ✅ 환경설정
load_dotenv()
st.set_page_config(page_title="🚗 현대차 vs 타제조사 비교", layout="wide")
st.title("🚗 Claude 기반 현대차 vs 타제조사 차량 비교")

ROOT_DIR = "C:/gamin/Project_12_2/hyundaicar_info"
VECTOR_DIR = "C:/gamin/Project_12_2/vector_db/price"

# ✅ 현대차 모델 목록
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
model_names = [name for _, name in car_model_map]

# ✅ UI 구성
col1, col2 = st.columns(2)

with col1:
    selected_model = st.selectbox("✅ 비교할 현대차 선택", ["선택 안함"] + model_names)

with col2:
    uploaded_pdf = st.file_uploader("📤 타제조사 가격표 PDF 업로드 (기아 등)", type=["pdf"])

question = st.text_input("💬 비교할 질문을 입력하세요 (예: 두 차량의 가격 차이는 얼마인가요?)")

# ✅ 벡터스토어 로딩 함수 (현대차용 FAISS)
@st.cache_resource
def load_hyundai_vectorstore(pdf_path, save_path):
    embeddings = OpenAIEmbeddings()
    if os.path.exists(os.path.join(save_path, "index.faiss")):
        return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    else:
        docs = PyPDFLoader(pdf_path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vectordb = FAISS.from_documents(chunks, embeddings)
        vectordb.save_local(save_path)
        return vectordb

# ✅ 실시간 PDF 벡터 임베딩 (Chroma 사용)
def create_temp_vectorstore_from_pdf(uploaded_file):
    if uploaded_file is None:
        return None, []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    docs = PyPDFLoader(tmp_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(chunks, embeddings)
    return vectordb, chunks

# ✅ Claude RAG 질의응답 체인
def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = ChatAnthropicMessages(model="claude-3-5-sonnet-20240620", temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        너는 차량 가격표 정보를 분석하는 자동차 전문가야.
        문서를 기반으로 정확한 답변을 제공해줘.

        질문: {question}
        문서 정보:
        {context}
        답변:
        """
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

# ✅ 질의응답 수행
if selected_model != "선택 안함" and uploaded_pdf and question:
    st.info("🔍 차량 정보 비교를 시작합니다...")

    # 현대차 vectordb
    rel_path, model_name = next((p, n) for p, n in car_model_map if n == selected_model)
    hyundai_pdf_path = os.path.join(ROOT_DIR, rel_path, f"{model_name}-price.pdf")
    hyundai_vectordb = load_hyundai_vectorstore(hyundai_pdf_path, os.path.join(VECTOR_DIR, model_name))
    hyundai_chain = build_qa_chain(hyundai_vectordb)

    # 타제조사 vectordb + chunks
    competitor_vectordb, competitor_chunks = create_temp_vectorstore_from_pdf(uploaded_pdf)
    competitor_chain = build_qa_chain(competitor_vectordb)

    with st.spinner("🤖 Claude가 현대차 응답 생성 중..."):
        hyundai_answer = hyundai_chain.run(question)
    with st.spinner("🤖 Claude가 타제조사 응답 생성 중..."):
        competitor_answer = competitor_chain.run(question)

    # ✅ 결과 출력
    st.markdown("### ✅ 비교 결과")
    st.markdown(f"#### ❓ 질문: `{question}`")
    st.markdown("---")
    st.markdown(f"#### 🚗 현대차 `{model_name}` 응답:")
    st.write(hyundai_answer)
    st.markdown("---")
    st.markdown("#### 🏷️ 타제조사 차량 응답:")
    st.write(competitor_answer)

    # ✅ 참조한 PDF Chunk 보기
    with st.expander("📄 참조한 문서 보기 (RAG 기반)"):
        st.markdown("#### 문서 1 (price)")
        for i, chunk in enumerate(hyundai_vectordb.similarity_search(question, k=1), 1):
            st.code(chunk.page_content.strip(), language="markdown")

        st.markdown("#### 문서 2 (uploaded)")
        for i, chunk in enumerate(competitor_vectordb.similarity_search(question, k=1), 1):
            st.code(chunk.page_content.strip(), language="markdown")
