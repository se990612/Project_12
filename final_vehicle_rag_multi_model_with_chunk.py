# 🚘 차량 통합 비교 RAG 서비스 최종
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import pdfplumber

from langchain.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropicMessages
from langchain_openai import ChatOpenAI

# ✅ 초기 설정
load_dotenv()
st.set_page_config(page_title="🚘 차량 정보 비교 RAG", layout="wide")
st.title("🚘 차량 가격/카탈로그 기반 Claude & GPT RAG 통합 응답")

ROOT_DIR = "C:/Users/KDT13/kh0616/project_12/Project_12/hyundaicar_info"
VECTOR_DIR = "C:/Users/KDT13/kh0616/project_12/Project_12/vector_db"
CATALOG_DB = os.path.join(VECTOR_DIR, "catalog")
PRICE_DB = os.path.join(VECTOR_DIR, "price")

if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"):
    st.error("❌ .env에 ANTHROPIC_API_KEY 또는 OPENAI_API_KEY 설정이 필요합니다.")
    st.stop()

# ✅ 차량 목록 불러오기
@st.cache_resource
def get_all_car_models():
    models = []
    for category in os.listdir(ROOT_DIR):
        category_path = os.path.join(ROOT_DIR, category)
        if not os.path.isdir(category_path):
            continue  # 이미지 등 파일은 무시

        for model_dir in os.listdir(category_path):
            model_path = os.path.join(category_path, model_dir)
            if not os.path.isdir(model_path):
                continue  # 하위 폴더도 디렉터리만

            catalog_file = next((f for f in os.listdir(model_path) if f.endswith("-catalog.pdf")), None)
            price_file = next((f for f in os.listdir(model_path) if f.endswith("-price.pdf")), None)
            if catalog_file and price_file:
                base_name = catalog_file.replace("-catalog.pdf", "")
                rel_path = os.path.relpath(model_path, ROOT_DIR)
                models.append((model_dir, rel_path, base_name))
    return models


# ✅ PDF + 표 병합 커스텀 로더
def custom_loader_with_table(pdf_path):
    docs = PyMuPDFLoader(pdf_path).load()
    table_texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or not table[0]: continue
                header = table[0]
                for row in table[1:]:
                    if row and any(cell for cell in row):
                        clean_row = [cell.strip() if cell else "" for cell in row]
                        row_text = ", ".join(f"{k.strip()}: {v.strip()}" for k, v in zip(header, clean_row))
                        table_texts.append(row_text)
    if table_texts:
        docs.append(Document(page_content="\n".join(table_texts)))
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    return splitter.split_documents(docs)

# ✅ 벡터 DB 로드/생성
@st.cache_resource
def load_or_create_faiss(pdf_path, save_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_file = os.path.join(save_path, "index.faiss")
    pkl_file = os.path.join(save_path, "index.pkl")
    if os.path.exists(index_file) and os.path.exists(pkl_file):
        return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    else:
        docs = custom_loader_with_table(pdf_path)
        vectordb = FAISS.from_documents(docs, embeddings)
        os.makedirs(save_path, exist_ok=True)
        vectordb.save_local(save_path)
        return vectordb

# ✅ 예산 필터
st.markdown("### 💸 먼저, 예산을 설정해주세요!")
budget_range = st.slider("예산 범위 (만원)", 1000, 7000, (2500, 5000), step=100)

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

filtered = []
for name, start_price, max_discount in benefit_data:
    discount_price = start_price - max_discount
    if budget_range[0] <= start_price <= budget_range[1] or budget_range[0] <= discount_price <= budget_range[1]:
        filtered.append((name, start_price, max_discount, discount_price))

if filtered:
    df_filtered = pd.DataFrame(filtered, columns=["차량명", "시작가 (만원)", "최대 할인 (만원)", "혜택 적용가 (만원)"])
    st.markdown("### 📊 정렬 옵션 선택")
    sort_column = st.selectbox("정렬 기준", ["혜택 적용가 (만원)", "시작가 (만원)", "최대 할인 (만원)"])
    sort_order = st.radio("정렬 방식", ["오름차순", "내림차순"], horizontal=True)
    ascending = True if sort_order == "오름차순" else False
    df_filtered = df_filtered.sort_values(sort_column, ascending=ascending).reset_index(drop=True)
    st.success(f"🔍 예산 {budget_range[0]}~{budget_range[1]}만원 차량 {len(df_filtered)}개")
    st.dataframe(df_filtered, use_container_width=True)
else:
    st.warning("현재 예산에 맞는 차량이 없습니다.")

# ✅ 차량 선택
st.markdown("### 🚗 차량 선택")
car_model_map = get_all_car_models()
car_model_options = [display_name for display_name, _, _ in car_model_map]
selected_model = st.selectbox("차량을 선택하세요", ["선택 안함"] + car_model_options)

if selected_model != "선택 안함":
    result = next(((rel_path, base_name) for display_name, rel_path, base_name in car_model_map if display_name == selected_model), None)
    if result:
        rel_path, base_name = result
        st.success(f"✅ `{selected_model}` 문서 로드 완료")
    else:
        st.error("❌ 차량 경로를 찾을 수 없습니다.")
        st.stop()
else:
    st.stop()

# ✅ PDF 경로 결정 함수
def find_relevant_pdf(question, model_name, rel_path):
    price_keywords = ["가격", "할인", "비용", "금액", "출고가", "옵션가"]
    pdf_type = "price" if any(kw in question for kw in price_keywords) else "catalog"
    filename = f"{model_name}-{pdf_type}.pdf"
    save_path = os.path.join(PRICE_DB if pdf_type == "price" else CATALOG_DB, model_name)
    return os.path.join(ROOT_DIR, rel_path, filename), save_path

# ✅ RAG 응답 함수
def get_rag_answer(llm, vectordb, query, name=""):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
        차량 카탈로그/가격표 PDF를 분석해 정보를 정확하게 설명하세요.
        질문: {{question}}
        문서 정보: {{context}}
        답변:
        """
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt_template})
    return qa.run(query)

# ✅ 일반 응답 함수
def get_basic_answer(llm, question):
    return llm.invoke(question).content

# ✅ 참조 chunk 미리보기 함수
def get_context_chunks(vectordb, query):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    return docs

# 🔁 대화 기록 초기화
if "chat_log" not in st.session_state:
    st.session_state["chat_log"] = []

# ✅ 질문 입력
question = st.text_input("💬 차량에 대해 궁금한 질문을 입력하세요")

if question and selected_model != "선택 안함":
    with st.spinner("⏳ 벡터 DB 로드 중..."):
        pdf_path, vector_path = find_relevant_pdf(question, base_name, rel_path)
        vectordb = load_or_create_faiss(pdf_path, vector_path)

    with st.spinner("🧠 Claude & GPT 응답 생성 중..."):
        # Claude vs Claude+RAG
        claude_llm = ChatAnthropicMessages(model="claude-3-5-sonnet-20240620", temperature=0.3)
        claude_rag = get_rag_answer(claude_llm, vectordb, question, "claude_rag")
        claude_basic = get_basic_answer(claude_llm, question)

        # GPT vs GPT+RAG
        gpt_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        gpt_rag = get_rag_answer(gpt_llm, vectordb, question, "gpt_rag")
        gpt_basic = get_basic_answer(gpt_llm, question)

        # 참조 chunk
        context_chunks = get_context_chunks(vectordb, question)

    # 🔁 히스토리 저장
    st.session_state["chat_log"].append({
        "question": question,
        "claude_rag": claude_rag,
        "claude_basic": claude_basic,
        "gpt_rag": gpt_rag,
        "gpt_basic": gpt_basic,
        "chunks": context_chunks
    })

# ✅ 전체 대화 기록 보기
for i, log in enumerate(reversed(st.session_state["chat_log"])):
    st.markdown(f"### 🔎 질문 {len(st.session_state['chat_log']) - i}: `{log['question']}`")

    with st.expander("🤖 Claude (기본 vs RAG)", expanded=True):
        st.markdown("**Claude (기본):**")
        st.markdown(log["claude_basic"])
        st.markdown("**Claude + RAG:**")
        st.markdown(log["claude_rag"])

    with st.expander("🧠 GPT (기본 vs RAG)", expanded=False):
        st.markdown("**GPT (기본):**")
        st.markdown(log["gpt_basic"])
        st.markdown("**GPT + RAG:**")
        st.markdown(log["gpt_rag"])

    with st.expander("📄 참조한 PDF 문서 Chunk", expanded=False):
        for idx, doc in enumerate(log["chunks"]):
            st.markdown(f"**Chunk {idx+1}:**")
            st.code(doc.page_content[:1000], language="text")  # 너무 길 경우 앞부분만 표시
