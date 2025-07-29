# 웹 크롤링 셀레니엄

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
from fpdf import FPDF
import os

# ✅ 크롬 브라우저 설정
options = Options()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=options)
driver.get("https://www.hyundai.com/kr/ko/e/vehicles/monthly-benefit")
time.sleep(7)  # 페이지 로딩 대기

# ✅ HTML 파싱
soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

# ✅ 차량 정보 수집
vehicles = []
items = soup.select("li[data-v-797473c4]")  # 모든 차량 블록 선택

for item in items:
    try:
        name = item.select_one("span.name strong").get_text(strip=True)
        start_price = item.select_one("span.start").get_text(strip=True)
        sale_tag = item.select_one("div.list > div > p:nth-child(2) > strong")
        sale = sale_tag.get_text(strip=True) if sale_tag else "-"
        vehicles.append((name, start_price, sale))
    except Exception as e:
        continue

# ✅ PDF 생성
pdf = FPDF()
pdf.add_page()
pdf.add_font("Nanum", "", "NanumGothic.ttf", uni=True)
pdf.set_font("Nanum", size=13)

pdf.cell(0, 10, "현대차 이달의 구매 혜택", ln=True, align='C')
pdf.ln(5)

# 테이블 헤더
pdf.set_fill_color(220, 220, 220)
pdf.set_font("Nanum", size=12)
pdf.cell(60, 10, "차량명", border=1, align="C", fill=True)
pdf.cell(60, 10, "시작가", border=1, align="C", fill=True)
pdf.cell(60, 10, "최대할인", border=1, align="C", fill=True)
pdf.ln()

# 테이블 내용
for name, price, discount in vehicles:
    pdf.cell(60, 10, name, border=1)
    pdf.cell(60, 10, price, border=1)
    pdf.cell(60, 10, discount, border=1)
    pdf.ln()

# ✅ PDF 저장
pdf.output("현대차_이달의_혜택.pdf")
print("✅ PDF 파일로 저장 완료: 현대차_이달의_혜택.pdf")
