# 📄 pages/5_타제조사_정보비교.py 최종
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic, ChatAnthropicMessages
from langchain.chains import RetrievalQAWithSourcesChain
import tempfile
from langchain.prompts import PromptTemplate

# ✅ 환경설정
load_dotenv()
st.set_page_config(page_title="🚗 현대차 vs 타제조사 비교", layout="wide")
st.title("🚗 Claude & GPT 기반 현대차 vs 타제조사 차량 비교")

ROOT_DIR = "C:/_knudata/Project_12/hyundaicar_info"
VECTOR_DIR = "C:/_knudata/Project_12/vector_db/price"

# ✅ 차량 목록
@st.cache_resource
def get_all_car_models():
    models = []
    for category in os.listdir(ROOT_DIR):
        category_path = os.path.join(ROOT_DIR, category)
        if not os.path.isdir(category_path):
            continue  # ✅ 폴더가 아닌 경우 스킵 (ex: .avif 파일)

        for model in os.listdir(category_path):
            model_path = os.path.join(category_path, model)
            if os.path.isdir(model_path):  # ✅ 여기도 디렉터리 확인
                models.append((os.path.join(category, model), model))
    return models

car_model_map = get_all_car_models()
model_names = [name for _, name in car_model_map]

# ✅ UI
col1, col2 = st.columns(2)
with col1:
    selected_model = st.selectbox("✅ 비교할 현대차 선택", ["선택 안함"] + model_names)
with col2:
    uploaded_pdf = st.file_uploader("📤 타제조사 가격표 PDF 업로드 (기아 등)", type=["pdf"])

question = st.text_input("💬 비교할 질문을 입력하세요 (예: 두 차량의 가격 차이는 얼마인가요?)")

# ✅ 현대차 FAISS
@st.cache_resource
def load_vectorstore(pdf_path, save_path):
    embeddings = OpenAIEmbeddings()
    if os.path.exists(os.path.join(save_path, "index.faiss")):
        return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    else:
        docs = PyPDFLoader(pdf_path).load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200).split_documents(docs)
        vectordb = FAISS.from_documents(chunks, embeddings)
        vectordb.save_local(save_path)
        return vectordb

# ✅ 타제조사 Chroma
def create_temp_vectorstore(uploaded_file):
    if not uploaded_file: return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    docs = PyPDFLoader(path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200).split_documents(docs)
    return Chroma.from_documents(chunks, OpenAIEmbeddings())

# ✅ QA 체인
def build_chain(model_name, retriever, provider):
    llm = ChatOpenAI(model_name=model_name) if provider == "gpt" else ChatAnthropic(model=model_name)

    prompt_template = PromptTemplate(
        input_variables=["summaries", "question"],
        template="""당신은 차량 가격표 PDF를 분석하는 한국어 전문가입니다.
모든 응답은 반드시 한국어로 해주세요.

질문: {question}
문서 내용: {summaries}

한국어로 답변:
"""
    )

    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

# ✅ Direct QA
def direct_answer(model_name, question, provider):
    llm = ChatOpenAI(model_name=model_name) if provider == "gpt" else ChatAnthropic(model=model_name)
    return llm.invoke(question).content

# ✅ 비교 실행
if selected_model != "선택 안함" and uploaded_pdf and question:
    rel_path, model_name = next((p, n) for p, n in car_model_map if n == selected_model)
    hyundai_path = os.path.join(ROOT_DIR, rel_path, f"{model_name}-price.pdf")
    hyundai_vector = load_vectorstore(hyundai_path, os.path.join(VECTOR_DIR, model_name))
    competitor_vector = create_temp_vectorstore(uploaded_pdf)
    
    hyundai_retriever = hyundai_vector.as_retriever()
    competitor_retriever = competitor_vector.as_retriever()

    # ✅ Direct
    hyundai_gpt_direct = direct_answer("gpt-4o", question, "gpt")
    hyundai_claude_direct = direct_answer("claude-3-5-sonnet-20240620", question, "claude")
    competitor_gpt_direct = direct_answer("gpt-4o", question, "gpt")
    competitor_claude_direct = direct_answer("claude-3-5-sonnet-20240620", question, "claude")

    # ✅ RAG
    hyundai_gpt_rag = build_chain("gpt-4o", hyundai_retriever, "gpt").invoke({"question": question})
    hyundai_claude_rag = build_chain("claude-3-5-sonnet-20240620", hyundai_retriever, "claude").invoke({"question": question})
    competitor_gpt_rag = build_chain("gpt-4o", competitor_retriever, "gpt").invoke({"question": question})
    competitor_claude_rag = build_chain("claude-3-5-sonnet-20240620", competitor_retriever, "claude").invoke({"question": question})

    st.markdown(f"### ❓ 질문: {question}")

    st.subheader("🔍 기본 LLM 응답 (PDF 미참조)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### 🚗 현대차 GPT-4o")
        st.info(hyundai_gpt_direct)
    with col2:
        st.markdown(f"#### 🏷️ 타제조사 GPT-4o")
        st.info(competitor_gpt_direct)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"#### 🚗 현대차 Claude")
        st.info(hyundai_claude_direct)
    with col4:
        st.markdown(f"#### 🏷️ 타제조사 Claude")
        st.info(competitor_claude_direct)

    st.markdown("### 📄 RAG 기반 응답 (PDF 기반)")
    col5, col6 = st.columns(2)
    with col5:
        st.markdown(f"#### 🚗 현대차 GPT-4o + RAG")
        st.success(hyundai_gpt_rag["answer"])
        st.caption(f"출처: {hyundai_gpt_rag.get('sources', 'N/A')}")
    with col6:
        st.markdown(f"#### 🏷️ 타제조사 GPT-4o + RAG")
        st.success(competitor_gpt_rag["answer"])
        st.caption(f"출처: {competitor_gpt_rag.get('sources', 'N/A')}")

    col7, col8 = st.columns(2)
    with col7:
        st.markdown(f"#### 🚗 현대차 Claude + RAG")
        st.success(hyundai_claude_rag["answer"])
        st.caption(f"출처: {hyundai_claude_rag.get('sources', 'N/A')}")
    with col8:
        st.markdown(f"#### 🏷️ 타제조사 Claude + RAG")
        st.success(competitor_claude_rag["answer"])
        st.caption(f"출처: {competitor_claude_rag.get('sources', 'N/A')}")


        # ✅ 상위 관련 문서 보기 함수
    def get_top_chunks(query, retriever, k=5):
        return retriever.get_relevant_documents(query)[:k]

    # ✅ 참조 문서 출력
    st.markdown("### 📚 참조한 PDF 문서 보기 (RAG 기반)")
    with st.expander("🚗 현대차 참조 문서 보기"):
        hyundai_chunks = get_top_chunks(question, hyundai_retriever)
        for i, doc in enumerate(hyundai_chunks, 1):
            st.markdown(f"**문서 {i} ({doc.metadata.get('source', 'unknown')})**")
            st.code(doc.page_content.strip(), language="markdown")

    with st.expander("🏷️ 타제조사 참조 문서 보기"):
        competitor_chunks = get_top_chunks(question, competitor_retriever)
        for i, doc in enumerate(competitor_chunks, 1):
            st.markdown(f"**문서 {i} ({doc.metadata.get('source', 'unknown')})**")
            st.code(doc.page_content.strip(), language="markdown")