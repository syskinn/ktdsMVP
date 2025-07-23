#시각화 및 웹앱 구성
import streamlit as st
import altair as alt
# 데이터 및 파일 처리
import pandas as pd
import openai
import os
import io
# 환경변수 및 유틸리티
from dotenv import load_dotenv
from collections import Counter
# 타입 선언 및 데이터 모델링
from typing import List, Literal, Dict
from pydantic import BaseModel, Field
# LangChain 및 OpenAI 연동
from langchain.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from defect_classification import defect_classify

# 환경변수 로드
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("OPENAI_AZURE_ENDPOINT")

# LLM 설정
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1",
    api_version="2024-12-01-preview",
    azure_endpoint=AZURE_ENDPOINT,
    temperature=0.0,
    api_key=AZURE_OPENAI_API_KEY
)

# 메인화면 Streamlit 설정
st.markdown(
    """
    <div style="text-align:center; padding:10px; border:2px solid #4CAF50; border-radius:10px;">
        <h2> 📊 KT 스마트폰 상용품질 모니터링</h2>
        <h3> [ VOC 감정분석 / 결함유형 / 주요키워드 ]</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# --- VOC 감정분석 정의 ---
class Sentiment(BaseModel):
    sentiment: Literal["긍정", "부정"] = Field(description="sentiment of a sentence")

system_template = """\
당신은 주어진 문장의 감정을 '긍정'인지 '부정'인지 분류하는 감정 분석기입니다.
아래 예시를 참고하여 새로운 문장의 감정을 정확히 판단하세요:
{examples}
"""

prompt_sentiment = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])

senti_chain = prompt_sentiment | llm.with_structured_output(Sentiment)

def senti_classify(user_input: str, examples: List[Dict[str, str]]) -> str:
    try:
        examples_str = "\n\n".join(
            f"문장: {ex['text']}\n감정: {ex['label']}"
            for ex in examples
        )
        result = senti_chain.invoke({
            "text": user_input,
            "examples": examples_str
        })
        return result.sentiment
    except Exception as e:
        return f"분석 실패: {str(e)}"

sample_examples = [
    {"text": "안녕하세요 현재 노트 페이지 모아보기 가 2행 기준인데 아래 처럼 4행이나 5행으로 바뀌면 좋겠어요 그럼 페이지 배열 수정하거나 한 눈에 보기 더 편할 거 같아요", "label": "긍정"},
    {"text": "안녕하세요 삼성노트에서 밑줄 그을때 직선 이외에도 물결 모양이나 점선 모양 밑줄 이 있으면 좋겠어요", "label": "긍정"},
    {"text": "갤럭시 s24 one ul8 안드로이드 16 언제 참여 가능할지 이번 언팩 유튜브 시청했는데 지금 워치7 울트라 디자인 좋타고 보네요", "label": "긍정"},
    {"text": "다음 갤럭시 s26 울트라 칩셋 퀄검 S팬 탑제 되길 바라네요 내년 갤럭시 언팩 유튜브 영상 기다릴게요", "label": "긍정"},
    {"text": "아싸! 손전등 하나 더 생겼네요~ 아오 *** 씬나~", "label": "긍정"},
    {"text": "울트라 쓰시는 분들 oul8 베타버전 어떠신가요?", "label": "긍정"},
    {"text": "이 서비스 정말 편리하고 만족스러워요.", "label": "긍정"},
    {"text": "앱이 자꾸 렉 걸려서 사용하기 불편했어요.", "label": "부정"},
    {"text": "[S24U] 업데이트 후 극심한 발열", "label": "부정"},
    {"text": "OneUI 7.0 업데이트 후 AI지우개 기능을 쓸 수 없게 됨 (S22)", "label": "부정"},
    {"text": "7.0 업데이트 이후 알람 안울림", "label": "부정"},
    {"text": "One UI 7.0 업데이트 후 게이밍 허브랑 게임 저만 튕기나요", "label": "부정"},
    {"text": "S23FE 기종 폰 터치가 몇분뒤에 터치가 안돼고 먹통이 됩니다 도와주세여", "label": "부정"},
    {"text": "s23입니다. 경고 설정을 해도 경고 알림이 안 뜨는데 어떻게 하면 뜨게 할 수 있을까요? 차단 설정해 놓으면 차단은 잘 됩니다.", "label": "부정"},
    {"text": "AOD 항상표시로 해도 몇 초뒤 꺼짐", "label": "부정"},
    {"text": "S25 울트라 가끔 중간위에 버벅 거리던데 폰이 불량같은데 교환할려면 절차가 어떻게 되나요?", "label": "부정"},
    {"text": "S24 Ultra 사용자입니다. 제어센터가 간혹 두번째 이미지처럼 변하고 변한 부분은 터치가 안됩니다.", "label": "부정"},
    {"text": "now brief 다운받았는데 now bar에는 안뜨네요", "label": "부정"},
    {"text": "7.0업그레이드하고나서 알람설정 블루투스연결 상단바에 표시 안되어 무지 불편합니다. 예전으로 갈수없나요? 07. 10. 17:49", "label": "부정"},
    {"text": "위젯도 크기를 줄일 수 없습니다 업데이트 후 위젯들이 다 커져서 기존에 사용하던 홈 화면 레이아웃 다 깨져버리고 불편함만 늘어서 괜히 했다 싶네요", "label": "부정"},
    {"text": "4k 60 프레임 영상 재생하면 무조건이라 해도 좋을만큼 뚝뚝 끊기면서 버벅거리는데 저만 이런가요?", "label": "부정"},
    {"text": "S25 시계앱이 사라졌어요.", "label": "부정"},

]


# --- VOC 유형분석 정의 defect_classification 호출 ---


# --- VOC 키워드 추출 정의 ---
class KeywordExtraction(BaseModel):
    keywords: List[str] = Field(description="문장에서 추출된 품질 관련 핵심 키워드 리스트")

system_prompt_kw = """\
당신은 문장에서 품질 이슈 관련 핵심 키워드만 추출하는 시스템입니다.
다음 기준을 따르세요:
- '화면', '터치', '속도', '충전', '오류', '버벅임' 등 품질 관련 명확한 키워드만 추출
- 단말명, 브랜드명(예: 갤럭시, 버즈, S24)은 제외
- '좋다', '싫다', '불편하다' 등 감정적·추상적 표현은 제외
- 결과는 키워드만 담긴 리스트(List[str])로 반환
예시:
문장: "화면이 자꾸 꺼지고 터치가 안 먹혀요"
결과: ["화면", "꺼짐", "터치", "미작동"]
"""

prompt_kw = ChatPromptTemplate.from_messages([
    ("system", system_prompt_kw),
    ("user", "{text}")
])

keyword_chain = prompt_kw | llm.with_structured_output(KeywordExtraction)

def extract_keywords(user_input: str) -> List[str]:
    if not user_input.strip():
        return ["입력된 문장이 없습니다."]
    try:
        result = keyword_chain.invoke({"text": user_input})
        return result.keywords
    except openai.error.AuthenticationError:
        return ["추출 실패: API 키 인증 오류"]
    except openai.error.OpenAIError as e:
        return [f"추출 실패: OpenAI API 오류 - {str(e)}"]
    except Exception as e:
        return [f"추출 실패: 알 수 없는 오류 - {str(e)}"]


# --- Streamlit UI (사이드바 + 메인) ---
# Streamlit의 session_state를 활용하여 분석 결과를 저장
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None

st.sidebar.header("📂 파일 업로드")
uploaded_file = st.sidebar.file_uploader("엑셀(.xlsx/.xls) 파일을 업로드하세요", type=["xlsx", "xls"])

if uploaded_file:
    if st.session_state.processed_df is None:  # 분석이 아직 실행되지 않은 경우
        df = pd.read_excel(uploaded_file)

        # '내용' 필드를 'strReview'로 변경
        if "내용" in df.columns:
            df.rename(columns={"내용": "strReview"}, inplace=True)

        text_col = next(
            (col for col in df.columns if col.lower() == "strreview" or "text" in col.lower() or "의견" in col),
            None
        )

        if not text_col:
            st.error("❌ 텍스트 컬럼을 찾을 수 없습니다. 'strReview', 'text', '의견', '내용' 등을 컬럼명에 포함해주세요.")
        else:
            st.success(f"✅ 분석 대상 컬럼: **{text_col}**")

            with st.spinner("📊 분석 중..."):
                df["strEmotion"] = df[text_col].apply(lambda x: senti_classify(str(x), sample_examples))
                df["strMainCategory"] = df[text_col].apply(lambda x: defect_classify(str(x)))
                df["strKeyword"] = df[text_col].apply(lambda x: extract_keywords(str(x)))

            st.session_state.processed_df = df  # 분석 결과를 session_state에 저장
            st.success("✅ 분석 완료!")
    else:
        df = st.session_state.processed_df  # 기존 분석 결과를 불러옴

    # 결과 Preview
    st.subheader("🔍 분석 결과 미리보기")
    st.dataframe(df.head(10), use_container_width=True)

    # 감정 분포 차트
    st.subheader("😊 감정 분포")
    emo_counts = df["strEmotion"].value_counts().reset_index()
    emo_counts.columns = ['감정', '건수']
    emo_chart = alt.Chart(emo_counts).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X('감정:N', title='감정'),
        y=alt.Y('건수:Q', title='빈도수'),
        color=alt.Color('감정:N', legend=None, scale=alt.Scale(scheme='category10'))
    ).properties(width=400, height=300)
    emo_text = emo_chart.mark_text(
        align='center',
        baseline='bottom',
        dy=-5
    ).encode(text='건수:Q')
    st.altair_chart(emo_chart + emo_text, use_container_width=False)

    # 결함 유형 분포 차트
    st.subheader("🛠️ 결함유형 분포")
    defect_counts = df["strMainCategory"].value_counts().reset_index()
    defect_counts.columns = ['결함유형', '건수']
    defect_chart = alt.Chart(defect_counts).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X('결함유형:N', sort='-y', title='결함유형'),
        y=alt.Y('건수:Q', title='빈도수'),
        color=alt.Color('결함유형:N', legend=None, scale=alt.Scale(scheme='tableau10'))
    ).properties(width=400, height=300)
    defect_text = defect_chart.mark_text(
        align='center',
        baseline='bottom',
        dy=-5
    ).encode(text='건수:Q')
    st.altair_chart(defect_chart + defect_text, use_container_width=False)

    # 키워드 Top 5 표 출력
    st.subheader("🔑 키워드 Top 5")
    all_keywords = sum(df["strKeyword"], [])
    top5 = Counter(all_keywords).most_common(5)
    top5_df = pd.DataFrame(top5, columns=["키워드", "언급 수"])
    st.table(top5_df.style.set_properties(**{'font-size': '16px', 'text-align': 'center'}))

    # 파일 다운로드
    output = io.BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    st.download_button(
        label="📥 결과 엑셀 다운로드",
        data=output.getvalue(),
        file_name="분석결과.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("⬅️ 사이드바에서 엑셀 파일을 업로드해주세요.")