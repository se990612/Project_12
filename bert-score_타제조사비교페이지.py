from bert_score import score

# ✅ 샘플 데이터 (사용자가 제공한 답변)
ref_pdf = "2026 아반떼 Smart (스마트) 20,650,000 개별소비세 3.5% 적용 시 20,340,000"

gpt_plain = "죄송하지만, 2026년 모델의 차량 가격에 대한 정보는 현재 제공할 수 없습니다. 차량 가격은 시장 상황, 옵션, 세금 등 여러 요인에 따라 달라질 수 있습니다. 가장 정확한 정보는 현대자동차나 기아 공식 홈페이지 또는 딜러십을 통해 확인하실 수 있습니다."

gpt_rag = "2026 아반떼 가솔린 스마트의 판매가격은 20,650,000원입니다."

claude_plain = "죄송합니다만 2026년형 모델의 정확한 가격 정보는 아직 공개되지 않았습니다. 자동차 가격은 매년 변동될 수 있으며, 미래 모델의 정확한 가격은 출시 시점에 가까워져야 알 수 있습니다. 현재 판매되는 모델을 기준으로 대략적인 가격대를 알려드리면: 아반떼 가솔린 스마트 트림: 약 2,000만원 ~ 2,200만원 사이"

claude_rag = "2026 아반떼 가솔린의 스마트 트림 판매가격은 20,650,000원입니다."

# ✅ BERTScore 비교 함수
def compare_with_bertscore(cand, ref, lang='ko'):
    P, R, F1 = score([cand], [ref], lang=lang)
    return {"precision": round(P[0].item(), 4), "recall": round(R[0].item(), 4), "f1": round(F1[0].item(), 4)}

# ✅ 결과
print("[GPT + RAG vs GPT]:", compare_with_bertscore(gpt_rag, gpt_plain))
print("[GPT + RAG vs PDF]:", compare_with_bertscore(gpt_rag, ref_pdf))

print("[Claude + RAG vs Claude]:", compare_with_bertscore(claude_rag, claude_plain))
print("[Claude + RAG vs PDF]:", compare_with_bertscore(claude_rag, ref_pdf))
