import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropicMessages

# ✅ 환경 변수 로드
load_dotenv()
if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("❌ .env 파일에 ANTHROPIC_API_KEY가 설정되어 있지 않습니다.")
    st.stop()

# ✅ PDF 경로
ROOT_DIR = "C:/_knudata/hyundaicar_info"
BENEFIT_PDF_PATH = os.path.join(ROOT_DIR, "현대차_이달의_혜택.pdf")

# ✅ Streamlit 설정
st.set_page_config(page_title="🚗 현대차 Claude RAG 데모", layout="wide")
st.title("🚗 현대차 Claude 기반 RAG 데모")
st.caption("카탈로그 + 가격표 기반 RAG 응답과 Claude 기본 응답 비교")

# ✅ 이달의 혜택 표 데이터 (PDF 내용 하드코딩)
benefit_data = [
    ("그랜저", "3,798만원", "170만원"),
    ("그랜저 Hybrid", "4,354만원", "170만원"),
    ("아반떼", "2,034만원", "155만원"),
    ("아반떼 Hybrid", "2,523만원", "155만원"),
    ("쏘나타 디 엣지", "2,788만원", "250만원"),
    ("쏘나타 디 엣지 Hybrid", "3,232만원", "250만원"),
    ("코나", "2,446만원", "155만원"),
    ("코나 Hybrid", "2,955만원", "155만원"),
    ("베뉴", "1,956만원", "155만원"),
    ("디 올 뉴 팰리세이드", "4,383만원", "126만원"),
    ("디 올 뉴 팰리세이드 Hybrid", "4,968만원", "126만원"),
    ("투싼", "2,729만원", "250만원"),
    ("투싼 Hybrid", "3,205만원", "250만원"),
    ("싼타페", "3,492만원", "250만원"),
    ("싼타페 Hybrid", "3,870만원", "150만원"),
    ("스타리아 라운지", "3,780만원", "335만원"),
    ("스타리아 라운지 Hybrid", "4,110만원", "235만원"),
    ("스타리아", "2,847만원", "335만원"),
    ("스타리아 Hybrid", "3,433만원", "235만원"),
    ("스타리아 킨더", "3,643만원", "335만원"),
    ("스타리아 라운지 캠퍼", "7,094만원", "335만원"),
    ("스타리아 라운지 캠퍼 Hybrid", "7,436만원", "235만원"),
    ("스타리아 라운지 리무진", "5,911만원", "335만원"),
    ("스타리아 라운지 리무진 Hybrid", "6,241만원", "235만원"),
    ("더 뉴 아이오닉 6", "4,856만원", "780만원"),
    ("디 올 뉴 넥쏘", "7,643만원", "495만원"),
    ("아이오닉 5", "4,740만원", "600만원"),
    ("코나 Electric", "4,152만원", "685만원"),
    ("아이오닉 9", "6,715만원", "370만원"),
    ("ST1", "5,655만원", "475만원"),
    ("포터 II Electric", "4,325만원", "485만원"),
    ("아반떼 N", "3,360만원", "455만원"),
    ("아이오닉 5 N", "7,700만원", "780만원"),
    ("포터 II", "2,028만원", "185만원"),
]
benefit_df = pd.DataFrame(benefit_data, columns=["차량명", "시작가(~부터)", "최대할인"])

model_name_map = {
    "grandeur": "그랜저",
    "grandeur-hybrid": "그랜저 Hybrid",
    "avante": "아반떼",
    "avante-hybrid": "아반떼 Hybrid",
    "sonata-the-edge": "쏘나타 디 엣지",
    "sonata-the-edge-hybrid": "쏘나타 디 엣지 Hybrid",
    "kona": "코나",
    "kona-hybrid": "코나 Hybrid",
    "venue": "베뉴",
    "the-all-new-palisade": "디 올 뉴 팰리세이드",
    "the-all-new-palisade-hybrid": "디 올 뉴 팰리세이드 Hybrid",
    "tucson": "투싼",
    "tucson-hybrid": "투싼 Hybrid",
    "santafe": "싼타페",
    "santafe-hybrid": "싼타페 Hybrid",
    "staria-lounge": "스타리아 라운지",
    "staria-lounge-hybrid": "스타리아 라운지 Hybrid",
    "staria": "스타리아",
    "staria-hybrid": "스타리아 Hybrid",
    "staria-kinder": "스타리아 킨더",
    "staria-lounge-camper": "스타리아 라운지 캠퍼",
    "staria-lounge-camper-hybrid": "스타리아 라운지 캠퍼 Hybrid",
    "staria-lounge-limousine": "스타리아 라운지 리무진",
    "staria-lounge-limousine-hybrid": "스타리아 라운지 리무진 Hybrid",
    "the-new-ioniq6": "더 뉴 아이오닉 6",
    "the-all-new-nexo": "디 올 뉴 넥쏘",
    "ioniq5": "아이오닉 5",
    "kona-electric": "코나 Electric",
    "ioniq9": "아이오닉 9",
    "st1": "ST1",
    "porter2-electric": "포터 II Electric",
    "avante-n": "아반떼 N",
    "ioniq5-n": "아이오닉 5 N",
    "porter2": "포터 II",
}

# ✅ 이달의 혜택 보기
with st.expander("📄 이달의 구매 혜택 보기", expanded=False):
    st.dataframe(benefit_df, use_container_width=True)

def get_benefit_for_model(model_name_en: str, benefit_df: pd.DataFrame, name_map: dict) -> str:
    kor_name = name_map.get(model_name_en)
    if not kor_name:
        return "❌ 해당 차량의 한글 모델명이 매핑되지 않았습니다."
    
    matched = benefit_df[benefit_df["차량명"] == kor_name]
    if matched.empty:
        return f"🔍 `{kor_name}` 차량에 대한 혜택 정보가 없습니다."
    
    row = matched.iloc[0]
    return f"💸 **{kor_name}**\n\n👉 시작가: `{row['시작가(~부터)']}`\n👉 최대 할인: `{row['최대할인']}`"


# ✅ 전체 차량 탐색
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

# ✅ Vectorstore 로딩
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

# ✅ Claude 응답
def answer_with_claude(vectorstore, query):
    retriever = vectorstore.as_retriever()
    llm = ChatAnthropicMessages(
        model="claude-3-5-sonnet-20240620",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(query)

# ✅ 차량 1개 선택: Claude 기본 + Hyundai RAG 비교
if len(selected_models) == 1:
    model_rel_path, model_name = next((fp, dn) for fp, dn in car_model_map if dn == selected_models[0])

    # ✅ 이달의 혜택 표시
    with st.expander("💸 선택 차량 이달의 혜택", expanded=True):
        benefit_text = get_benefit_for_model(model_name, benefit_df, model_name_map)
        st.markdown(benefit_text)

    base_path = os.path.join(ROOT_DIR, model_rel_path)
    catalog_path = os.path.join(base_path, f"{model_name}-catalog.pdf")
    price_path = os.path.join(base_path, f"{model_name}-price.pdf")

    if not os.path.exists(catalog_path) or not os.path.exists(price_path):
        st.error("❌ 카탈로그 또는 가격표 PDF가 없습니다.")
    else:
        vectordb = load_vectorstore_combined([catalog_path, price_path])

        st.subheader(f"🚘 [{model_name}] 차량 질문")
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

            st.markdown("---")
            st.markdown(f"### 🚘 차량: `{model_name}` | ❓ 질문: `{question}`")

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
