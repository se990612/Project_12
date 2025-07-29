# 📦 필수 라이브러리 임포트
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# ✅ .env 로드 및 페이지 설정
load_dotenv()
st.set_page_config(page_title="📘 Claude & GPT 차량 질의응답", layout="wide")
st.title("📘 Claude & GPT 기반 차량 카타로그 + 가격표 통합 RAG 질문응답")

ROOT_DIR = "C:/gamin/Project_12_2/hyundaicar_info"
VECTORSTORE_DIR = "C:/gamin/Project_12_2/vector_db"

# ✅ API 키 확인
if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"):
    st.error("❌ .env에 ANTHROPIC_API_KEY 또는 OPENAI_API_KEY가 누락되어 있습니다.")
    st.stop()

# ✅ 차량 폴더 구조 탐색
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

# ✅ 통합 벡터DB 생성 or 불러오기 (OpenAI 임베딩 + 참조 테귵 포함)
@st.cache_resource
def load_or_create_combined_vectorstore(catalog_path, price_path, save_path):
    embeddings = OpenAIEmbeddings()
    index_file = os.path.join(save_path, "index.faiss")
    pkl_file = os.path.join(save_path, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(pkl_file):
        return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    catalog_docs = PyPDFLoader(catalog_path).load()
    catalog_chunks = splitter.split_documents(catalog_docs)
    for doc in catalog_chunks:
        doc.metadata["source"] = "catalog"

    price_docs = PyPDFLoader(price_path).load()
    price_chunks = splitter.split_documents(price_docs)
    for doc in price_chunks:
        doc.metadata["source"] = "price"

    all_chunks = catalog_chunks + price_chunks
    vectordb = FAISS.from_documents(all_chunks, embeddings)
    os.makedirs(save_path, exist_ok=True)
    vectordb.save_local(save_path)
    return vectordb

# ✅ 질의응답 체인 생성
def build_qa_chain(model_name, retriever, provider="gpt"):
    if provider == "gpt":
        llm = ChatOpenAI(model_name=model_name, temperature=0)
    elif provider == "claude":
        llm = ChatAnthropic(model=model_name, temperature=0)
    else:
        raise ValueError("지원되지 않는 모델 제공자")
    return RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)

# ✅ 상위 관련 문서 가져오기
def get_top_chunks(query, retriever, k=5):
    return retriever.get_relevant_documents(query)[:k]

# ✅ Streamlit UI 시작
car_model_map = get_all_car_models()
car_model_options = [model_name for _, model_name in car_model_map]
selected_model = st.selectbox("🚗 차량 선택", car_model_options)

if selected_model:
    rel_path, model_name = next((fp, dn) for fp, dn in car_model_map if dn == selected_model)
    catalog_path = os.path.join(ROOT_DIR, rel_path, f"{model_name}-catalog.pdf")
    price_path = os.path.join(ROOT_DIR, rel_path, f"{model_name}-price.pdf")

    if not os.path.exists(catalog_path):
        st.error("❌ 카타로그 PDF가 존재하지 않습니다.")
    elif not os.path.exists(price_path):
        st.error("❌ 가격표 PDF가 존재하지 않습니다.")
    else:
        st.success(f"📄 `{model_name}`의 카타로그 + 가격표 로드 완료")

        question = st.text_input("❓ 차량에 대해 궁금한 점을 입력하세요 (예: 가격, 옵션, 연비 등)")

        if question:
            with st.spinner("📅 벡터스톤 로드 중..."):
                vectordb_path = os.path.join(VECTORSTORE_DIR, model_name)
                vectordb = load_or_create_combined_vectorstore(catalog_path, price_path, vectordb_path)
                retriever = vectordb.as_retriever()

            with st.spinner("🤖 GPT & Claude 응답 생성 중..."):
                gpt_chain = build_qa_chain("gpt-4o", retriever, provider="gpt")
                gpt_result = gpt_chain.invoke({"question": question})

                claude_chain = build_qa_chain("claude-3-haiku-20240307", retriever, provider="claude")
                claude_result = claude_chain.invoke({"question": question})

                top_chunks = get_top_chunks(question, retriever, k=5)

            # ✅ 결과 출력
            st.markdown(f"### ✅ 질문: {question}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("🧠 **GPT 응답**")
                st.success(gpt_result["answer"])
                st.caption(f"📚 출처: {gpt_result.get('sources', 'N/A')}")
            with col2:
                st.markdown("🧠 **Claude 응답**")
                st.success(claude_result["answer"])
                st.caption(f"📚 출처: {claude_result.get('sources', 'N/A')}")

            with st.expander("📄 참조한 문서 보기"):
                for i, doc in enumerate(top_chunks, 1):
                    st.markdown(f"**문서 {i} ({doc.metadata.get('source', 'unknown')})**")
                    st.code(doc.page_content.strip(), language="markdown")
