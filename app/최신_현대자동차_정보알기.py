import streamlit as st
import os
from PIL import Image

st.set_page_config(page_title="🚗 현대차 정보 서비스", layout="wide")

# ✅ 서비스 제목 및 설명
st.title("🚘 현대자동차 Claude & GPT 기반 AI 정보 서비스")
st.caption("공식 카탈로그 · 가격표 · 이달의 혜택 PDF 기반의 고품질 AI 차량 정보 Q&A")

# ✅ 대표 이미지 삽입
REPRESENTATIVE_IMG_PATH = "대표이미지.jpg"  # 한 장만 표시
if os.path.exists(REPRESENTATIVE_IMG_PATH):
    st.image(REPRESENTATIVE_IMG_PATH, use_container_width=True)

st.markdown("""
### 🧠 서비스 소개

**현대차 AI 정보 서비스**는 기존의 단순 GPT 응답을 넘어, 실제 차량 카탈로그와 가격표 PDF를 바탕으로 **정확하고 신뢰도 높은 차량 정보**를 제공하는 AI Q&A 서비스입니다.

📌 **주요 기능**
- 🔍 **Claude vs GPT 모델 비교 응답 제공**  
   └ 동일한 질문에 대해 **GPT-4o, Claude, GPT+RAG, Claude+RAG** 각각의 응답을 한눈에 비교  
   └ RAG 기반 답변은 실제 PDF 내용을 근거로 제공되어 신뢰도↑

- 📂 **PDF 기반 고품질 답변 (RAG)**  
   └ 차량 가격표·카탈로그·혜택 문서를 벡터로 변환 → 문서 기반으로 질문 검색  

- 💸 **예산 필터링 기반 차량 추천**  
   └ 차량 가격표를 기반으로 실시간 가격 조건 반영 차량 추천

- 🧾 **응답 근거 PDF 원문 확인 기능 제공**  
   └ “이 정보는 어느 문서에서 나왔는지?” → 하단 ‘📎 참조 PDF 문서 보기’로 확인 가능

> ✅ 기존 GPT 모델의 추측성 답변을 넘어, **실제 PDF 기반으로 정확한 응답**을 제공합니다.
""")

st.markdown("---")
st.markdown("👉 좌측 사이드바에서 원하는 기능(차량 Q&A, 비교 분석, PDF 기반 추천 등)을 체험해보세요!")
