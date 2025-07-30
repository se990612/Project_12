from bert_score import score

# ✅ 샘플 데이터 (사용자가 제공한 답변)
ref_pdf = "디 올 뉴 팰리세이드 하이브리드 Exclusive(익스클루시브) 49,820,000"

gpt_plain = "죄송하지만, 디 올 뉴 팰리세이드 하이브리드의 익스클루시브 등급 판매가격에 대한 구체적인 정보를 제공할 수 없습니다. 자동차의 가격은 국가, 지역, 옵션 및 세금에 따라 달라질 수 있으므로, 가장 정확한 정보는 현대자동차의 공식 웹사이트나 가까운 대리점을 통해 확인하시기 바랍니다."

gpt_rag = "디 올 뉴 팰리세이드 하이브리드의 익스클루시브 등급 판매가격은 49,820,000원입니다."

claude_plain = "디 올 뉴 팰리세이드 하이브리드의 익스클루시브 등급의 정확한 판매가격은 현재 공식적으로 발표되지 않았습니다. 현대자동차는 2023년 11월 팰리세이드 하이브리드 모델을 출시할 예정이라고 밝혔지만, 구체적인 가격 정보는 아직 공개하지 않았습니다. 일반적으로 하이브리드 모델은 동급 가솔린 모델보다 약 250-300만원 정도 비싼 편입니다. 따라서 기존 팰리세이드 가솔린 모델의 익스클루시브 트림 가격을 참고하면 대략적인 예상 가격을 추정해볼 수 있습니다. 정확한 가격은 출시 시점에 현대자동차 공식 홈페이지나 전시장을 통해 확인하실 수 있을 것입니다."

claude_rag = "디 올 뉴 팰리세이드 하이브리드의 익스클루시브 등급 판매가격은 49,820,000원입니다."

# ✅ BERTScore 비교 함수
def compare_with_bertscore(cand, ref, lang='ko'):
    P, R, F1 = score([cand], [ref], lang=lang)
    return {"precision": round(P[0].item(), 4), "recall": round(R[0].item(), 4), "f1": round(F1[0].item(), 4)}

# ✅ 결과
print("[GPT + RAG vs GPT]:", compare_with_bertscore(gpt_rag, gpt_plain))
print("[GPT + RAG vs PDF]:", compare_with_bertscore(gpt_rag, ref_pdf))

print("[Claude + RAG vs Claude]:", compare_with_bertscore(claude_rag, claude_plain))
print("[Claude + RAG vs PDF]:", compare_with_bertscore(claude_rag, ref_pdf))
