import os
import json
from textwrap import dedent
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from pydantic import BaseModel, Field

load_dotenv()

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

api_version="2024-12-01-preview"
llm_dep_name = "gpt-4.1"
embed_dep_name = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
openai_endpoint = os.getenv("OPENAI_AZURE_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("OPENAI_API_KEY")

azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

top_k = 5  # llm 이 판단할 때, 참고하는 예시 샘플의 수

# --- VOC 유형분석 정의 ---
def read_jsonl(file_path: str) -> list[dict]:
    """
    JSONL 파일을 읽어 각 줄을 JSON 객체로 파싱한 뒤 리스트로 반환합니다.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 빈 줄 스킵
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            data.append(record)
    return data


DEFECT_TYPES = Literal[
    "UI/UX", "기능/동작", "데이터/서버", "설치/업데이트", "성능 및 안정성", "멀티미디어",
    "네트워크", "호환성", "인증/권한/알림", "다국어/번역", "규격", "메시지/통화",
    "파일 처리 관련 결함", "시스템/데이터", "위치모드", "사용자 경험", "하드웨어",
    "외부기기 연동", "기타"
]


class DefectClass(BaseModel):
    defect_type: DEFECT_TYPES = Field(description="결함유형")


class KtDsSemanticSimilarityExampleSelector(SemanticSimilarityExampleSelector):
    def _documents_to_examples(self, documents: list[Document]) -> list[dict]:
        # Get the examples from the metadata.
        # This assumes that examples are stored in metadata.
        examples = [{**dict(e.metadata), "content": e.page_content} for e in documents]
        # If example keys are provided, filter examples to those keys.
        if self.example_keys:
            examples = [{k: eg[k] for k in self.example_keys} for eg in examples]
        return examples

    
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=embed_dep_name,
    openai_api_version=api_version,
    azure_endpoint=openai_endpoint,
    api_key=AZURE_OPENAI_KEY,
)

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=azure_search_endpoint, # ai search 서비스의 엔드포인트
    azure_search_key=AZURE_SEARCH_KEY, # ai search 서비스의 키
    index_name=search_index_name,
    search_type="hybrid", # hybrid 가 기본 값이다. 가능한 값: 'similarity', 'similarity_score_threshold', 'hybrid', 'hybrid_score_threshold', 'semantic_hybrid', 'semantic_hybrid_score_threshold'
    embedding_function=embeddings.embed_query,
    additional_search_client_options={"retry_total": 3, "api_version":"2025-05-01-preview"},
)


llm = AzureChatOpenAI(
    azure_deployment=llm_dep_name,
    api_version=api_version,
    azure_endpoint=openai_endpoint,
    temperature=0.,
    api_key=AZURE_OPENAI_KEY
)

defect_classify_llm = llm.with_structured_output(DefectClass)
example_selector = KtDsSemanticSimilarityExampleSelector(
    vectorstore=vector_store,
    k=top_k
)


# 결함 판단 기준
defect_guides = read_jsonl("defect.jsonl")

# 결함 판단 프롬프트
sys_prompt = dedent(
"""
당신은 결함유형을 판단하는 전문 엔지니어입니다. 

판단의 유형은 다음 가이드 라인을 따릅니다.
결함 유형과 해당 결함 유형의 내용입니다.

[판단 기준]
{% for i in range(DEFECT_GUIDES | length) %}
    - DEFECT_LABEL: {{ DEFECT_GUIDES[i]['label'] }} CASE: {{ DEFECT_GUIDES[i]['text'] }}
{% endfor %}

[지켜야할 사항]
- 총 결함 유형은 18가지이고, 가장 적절한 결함유형을 하나만 정확히 선택하세요.

결함 유형 판단은 아래를 참고하세요

[결함 유형 판단 예시]

""")

defect_clf_prompt = ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    MessagesPlaceholder("few_shots"),
    ("human", "다음 사용자의 메세지와 관련있는 결함 유형에대해서 찾아라\n target: {{input}}")
],  template_format="jinja2")


defect_classify_pipeline = defect_clf_prompt | defect_classify_llm


# Few-shot 프롬프트 템플릿
few_shot_prompt_template = ChatPromptTemplate.from_messages([
    ("human", "{content}"),
    ("assistant", "{strMainCategory}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=few_shot_prompt_template,
    example_selector=example_selector,
    input_variables=["input"]         
)


def defect_classify(query):
    few_shots = few_shot_prompt.invoke(query).to_messages()

    try:
        result = defect_classify_pipeline.invoke(
            {
                "input": query,
                "few_shots": few_shots,
                "DEFECT_GUIDES": defect_guides
            }
                
        )
    except Exception as e:
        print(f"[defect_classify] 에러 발생")
        return None
    else:
        return result.defect_type


if __name__ == "__main__":
    query = "약 1달쯤 전부터 모든 앱의 알림이 최소 5분에서 길면 20분가량 늦게 옵니다"
    few_shots = few_shot_prompt.invoke(query).to_messages()
    results = defect_classify_pipeline.invoke(
        {
            "input": query,
            "few_shots": few_shots,
            "DEFECT_GUIDES": defect_guides
        }
            
    )

    print(f"사용자 문의 사항: {query}")
    print(f"LLM 의 판단: {results.defect_type}")
    print("\n\n\n")
    print(f"판단할 때 참고한 샘플들\n")
    for fw in few_shots:
        if fw.type == 'human':
            print(f"문의: {fw.content}")
        else:
            print(f"정답: {fw.content}")
            print("-"*50)