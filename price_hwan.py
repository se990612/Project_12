# 📄 pages/3_가격표_QnA.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropicMessages
from langchain.prompts import PromptTemplate
import pandas as pd

#fd

# ✅ 환경설정
load_dotenv()
st.set_page_config(page_title="💵 가격표 Q&A", layout="wide")
st.title("💵 차량 가격표 기반 Claude RAG 질의응답")

ROOT_DIR = "C:/Users/KDT13/kh0616/project_12/Project_12/hyundaicar_info"
VECTORSTORE_DIR = "C:/Users/KDT13/kh0616/project_12/Project_12/vector_db/price"

# ✅ 차량 탐색 함수
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
st.markdown("### 🎯 옵션 1: 특정 차종에 대해 가격 질문하기")
selected_model = st.selectbox("🚗 차량 선택", ["선택 안함"] + car_model_options)

# ✅ FAISS 저장/불러오기
@st.cache_resource
def load_or_create_faiss(pdf_path, save_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    index_file = os.path.join(save_path, "index.faiss")
    pkl_file = os.path.join(save_path, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(pkl_file):
        return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    else:
        docs = PyPDFLoader(pdf_path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vectordb = FAISS.from_documents(chunks, embeddings)

        os.makedirs(save_path, exist_ok=True)
        vectordb.save_local(save_path)
        return vectordb

# ✅ Claude 응답 함수
def answer_with_claude(vectordb, query):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    llm = ChatAnthropicMessages(model="claude-3-5-sonnet-20240620", temperature=0)

    # ✅ 일반 LLMChain용 프롬프트로 변경
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    너는 차량 카탈로그 PDF를 기반으로 정보를 추출하는 전문가야.
    연비, 디자인, 옵션, 가격 등을 명확하게 설명하고, 반드시 문서 기반 사실만 언급해.

    질문: {question}
    문서 정보:
    {context}
    답변:
    """
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

    return qa.run(query)

# ✅ 차종 기반 질문
if selected_model != "선택 안함":
    rel_path, model_name = next((fp, dn) for fp, dn in car_model_map if dn == selected_model)
    price_path = os.path.join(ROOT_DIR, rel_path, f"{model_name}-price.pdf")

    if not os.path.exists(price_path):
        st.error("❌ 가격표 PDF가 존재하지 않습니다.")
    else:
        st.success(f"📄 {model_name} 가격표 로드 완료")
        question = st.text_input("💬 해당 차량에 대해 궁금한 가격 질문을 입력하세요")

        if question:
            with st.spinner("Claude RAG 응답 생성 중..."):
                faiss_dir = os.path.join(VECTORSTORE_DIR, model_name)
                vectordb = load_or_create_faiss(price_path, faiss_dir)
                response = answer_with_claude(vectordb, question)

            st.markdown("---")
            st.markdown(f"### 🚗 차량: `{model_name}`")
            st.markdown(f"### ❓ 질문: `{question}`")
            st.markdown("---")
            st.markdown("### 🤖 Claude RAG 응답")
            st.write(response)

# ✅ 예산 기반 필터링 기능
st.markdown("### 💸 옵션 2: 내 예산에 맞는 차량 추천")
budget_range = st.slider("예산 범위를 선택하세요", 1000, 10000, (2500, 5000), step=100)

benefit_data = [
    ("그랜저", 3798, 170), ("그랜저 Hybrid", 4354, 170), ("아반떼", 2034, 155), ("아반떼 Hybrid", 2523, 155),
    ("쏘나타 디 엣지", 2788, 250), ("쏘나타 디 엣지 Hybrid", 3232, 250), ("코나", 2446, 155), ("코나 Hybrid", 2955, 155),
    ("베뉴", 1956, 155), ("디 올 뉴 팰리세이드", 4383, 126), ("디 올 뉴 팰리세이드 Hybrid", 4968, 126),
    ("투싼", 2729, 250), ("투싼 Hybrid", 3205, 250), ("싼타페", 3492, 250), ("싼타페 Hybrid", 3870, 150),
    ("스타리아 라운지", 3780, 335), ("스타리아 라운지 Hybrid", 4110, 235), ("스타리아", 2847, 335),
    ("스타리아 Hybrid", 3433, 235), ("스타리아 킨더", 3643, 335), ("스타리아 라운지 캠퍼", 7094, 335),
    ("스타리아 라운지 캠퍼 Hybrid", 7436, 235), ("스타리아 라운지 리무진", 5911, 335),
    ("스타리아 라운지 리무진 Hybrid", 6241, 235), ("더 뉴 아이오닉 6", 4856, 780), ("디 올 뉴 넥쏘", 7643, 495),
    ("아이오닉 5", 4740, 600), ("코나 Electric", 4152, 685), ("아이오닉 9", 6715, 370), ("ST1", 5655, 475),
    ("포터 II Electric", 4325, 485), ("아반떼 N", 3360, 455), ("아이오닉 5 N", 7700, 780), ("포터 II", 2028, 185)
]

if st.button("🚀 예산 적용 및 추천 차량 보기"):
    filtered = []
    for name, start_price, max_discount in benefit_data:
        discount_price = start_price - max_discount
        if budget_range[0] <= start_price <= budget_range[1] or budget_range[0] <= discount_price <= budget_range[1]:
            filtered.append((name, start_price, max_discount, discount_price))
    st.session_state["filtered_cars"] = pd.DataFrame(
        filtered, columns=["차량명", "시작가", "최대 할인", "혜택 적용가"]
    )

if "filtered_cars" in st.session_state and not st.session_state["filtered_cars"].empty:
    df_filtered = st.session_state["filtered_cars"]

    sort_column = st.selectbox("정렬 기준", ["시작가", "최대 할인", "혜택 적용가"])
    sort_order = st.radio("정렬 순서", ["오름차순", "내림차순"], horizontal=True)
    ascending = sort_order == "오름차순"

    df_filtered = df_filtered.sort_values(by=sort_column, ascending=ascending).reset_index(drop=True)

    st.success(f"예산 {budget_range[0]}만원 ~ {budget_range[1]}만원에 해당하는 차량:")
    st.dataframe(df_filtered, use_container_width=True)

    csv = df_filtered.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 CSV로 저장하기",
        data=csv,
        file_name="예산별_추천_차량.csv",
        mime="text/csv"
    )
