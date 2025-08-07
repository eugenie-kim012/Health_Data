import os
import streamlit as st
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    client = OpenAI(api_key=api_key)
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.error("⚠️ .env 파일을 확인해 주세요: OPENAI_API_KEY를 설정하거나 유효한 값을 제공해야 앱이 정상 작동합니다.")

st.set_page_config(page_title="🧠 보건복지의료 빅데이터 연구 프로포절 작성 지원 튜터", layout="wide")
st.title("🧠 보건복지의료 빅데이터 연구 프로포절 작성 지원 튜터")

# --- 사이드바 (브레인스토밍 폼) ---
with st.sidebar:
    # Moved the st.form block directly inside st.sidebar
    with st.form("proposal_form"):
        st.subheader("프로포절 브레인스토밍 (Research Proposal Brainstorming)✨")

        field = st.text_input("1️⃣ 연구 분야 / 대상 (Research Field/ Subject)", placeholder="예: 보건 경제/ 국제 보건/ 역학", key="sidebar_field")
        topic = st.text_input("2️⃣ 연구 주제 (Research Topic)", placeholder="예: 구체적인 연구 아이디어", key="sidebar_topic")
        goal = st.text_area("3️⃣ 연구 목적 (Research Objective)", placeholder="이 연구의 목적은 무엇인가요?", key="sidebar_goal")
        method = st.selectbox("4️⃣ 분석 방법 (Methodology)", ["회귀분석", "머신러닝", "딥러닝", "시계열 분석", "LLM", "기타"], key="sidebar_method")
        dataset = st.text_input("5️⃣ 사용할 데이터셋 (Dataset to Be Used)", placeholder="예: 공공데이터", key="sidebar_dataset")
        institution = st.text_input("6️⃣ 지원 기관/학교 (Target Institution / Funding Body", placeholder="예: 한국연구재단", key="sidebar_institution")

        # st.form_submit_button must be inside st.form
        submitted = st.form_submit_button("✍️ 프로포절 초안 생성")

# --- 메인 콘텐츠 영역 ---
# 챗 메시지 기록 초기화 (세션 상태에 저장) - 메인 영역에 그대로 유지
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "tutor", "content": "💡빅데이터 기반 보건 연구 프로포절 초안 LLM과 같이 고민해봐요💡"}]

# 폼 제출 로직은 사이드바 안에 있지만, 결과 처리는 메인 영역에서 이루어집니다.
if submitted and api_key:
    if not all([field, topic, goal, dataset, institution]):
        st.warning("모든 필드를 채워주세요!")
    else:
        with st.spinner("프로포절 초안 작성 중입니다. 챗봇은 브레인스토밍 단계에서 사용하시고 멘토 및 튜터와 상의 후 추가 연구 조사를 직접 수행하시길 바랍니다. 행운을 빕니다🍀"):
            # 시스템 프롬프트 정의 (튜터의 역할 및 지침 - 프로포절 생성용)
            system_prompt_for_proposal_generation = """
[Role Definition]
You are a tutor specialized in assisting with the creation of big data-based research proposals. Your role is to provide friendly yet clear, technically sound, and domain-specific guidance, aiming to produce a detailed research proposal that could be presented to a mentor or tutor for further discussion.
Provide the outputs in paragraphs in very clear, technical and detailed sentences.

[Expected User Input]
The student will provide the following information:

1. Research Subject/Field: (e.g., Health economics, global health, Epidemiology)
2. Research Topic: (Specific research idea)
3. Research Objective: (What the research aims to achieve)
4. Preferred Analysis Method: (Preferred analytical techniques or ideas)
5. Dataset to be Used: (What data will be utilized)
6. Target Institution/University: (To infer or actively search for the institution's ethos and align the proposal accordingly, providing sources)

[Research Proposal Draft Guidelines]
Based on the student's input and considering the ethos of the target institution/university (after online search), generate a research proposal draft. If the student's initial direction is incorrect, provide friendly and clear corrective guidance. Each section should be academic and professional - however, provide those in paragraphs, not in bullet points:


1. Proposed Research Titles: Suggest 2-3 compelling titles that capture the essence of the research.
2. Background and Research Objectives/Necessity:
   - Briefly present the latest research trends in the field and identify the limitations of existing research or currently unresolved problems.
   - Clearly explain the specific problem this research aims to solve and the expected academic/practical impact.
   - Emphasize why this research is important at this time and its necessity.
   - Provide 3 URL links to actual, existing relevant academic papers found via Google Scholar for the student to review.
3. Research Questions and Hypothesis Direction:
   - Formulate 2-3 clear and verifiable research questions that the study will address.
   - Develop specific research hypotheses (tentative answers to the research questions) and outline their direction.
   - Clearly define the scope of the research to demonstrate its practical feasibility.
   - Provide 3 URL links to actual, existing similar academic papers found via Google Scholar.
4. Dataset Acquisition Strategy:
   - Provide detailed and technical advice, including an evaluation of the suitability of the student's proposed dataset, potential for acquiring additional data sources, and specific methods for data collection.
   - Offer practical guidelines for data preprocessing, cleaning, integration, and management.
   - Include advice on ethical considerations such as data security and personal information protection.
   - If the student's suggested method is unsuitable, recommend alternative methods as a tutor.
5. Data Analysis Methodology (Core):
   - Suggest how the techniques learned in the bootcamp (regression analysis, machine learning, deep learning, time series analysis, LLM applications) can be applied to this research realistically and concretely.
   - Clearly explain the rationale behind the choice of each analysis technique (e.g., specific machine learning algorithms, deep learning model architectures) and logically connect how it directly contributes to solving the research problem.
   - Include details on model evaluation metrics, interpretation of results, potential limitations, and proposed mitigation strategies.
   - If possible, suggest ways to combine multiple analysis techniques to enhance research depth and reliability.
   - If the student's proposed methodology is incorrect, recommend better methodologies based on professional search, providing sources.
6. Expected Contributions and Policy Implications:
   - Specifically describe what new knowledge this research will provide academically and how it will contribute to existing research.
   - Explicitly outline how the research findings can be utilized in policy formulation, social problem-solving, and industrial development, with concrete, real-world examples supported by accurate search results and sources.
   - Emphasize the social, economic, and public policy value of the research.
7. References:
   - Provide at least 3 accurate references, including URL links to actual, existing academic papers found via Google Scholar, formatted in APA style or a specified academic journal style, for any concepts or arguments mentioned.
8. Overall Assessment and Direction Review:
   - Conduct a precise analysis and search-based evaluation of the student's input, and as a tutor, advise on what additional tasks are required to complete the proposal.
9. Proposal Example:
   - Generate a detailed and technical example of a proposal, at least three paragraphs long, in the style of an academic paper or official report. This should be comprehensive enough to be presented directly to a tutor or mentor for discussion, including logical flow, evidence, citations, and complete references.
10. Closing remark:
   -  Please let them know that they should download the text-form data to avoid missing any information during further interactions with the chatbot.

[Tutor Feedback and Advice Style]
* Maintain a friendly yet clear tone, ensuring student comprehension.
* Explain technical aspects in a practical manner, considering the student's bootcamp background (avoiding overly complex jargon), but recommend additional learning resources if needed.
* Provide at least 3 accurate references based on precise searches.
* Present the methodology section in a detailed and technical manner.
* Provide accurate URL links to institutional/university Proposal guidelines based on Google search.
* Ensure the generated proposal draft is of high quality and completeness.
* If the student write his/her input in Korean, provide your input in Korean. 
"""
            user_prompt_for_proposal_generation = f"""
- 연구 분야/대상: {field}
- 연구 주제: {topic}
- 연구 목적: {goal}
- 분석 방법: {method}
- 사용할 데이터셋: {dataset}
- 지원 기관/학교: {institution}
"""
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt_for_proposal_generation},
                        {"role": "user", "content": user_prompt_for_proposal_generation}
                    ],
                    temperature=0.7
                )
                result = response.choices[0].message.content
                st.success("✅ 프로포절 초안이 생성되었습니다!")
                st.markdown(result)

                # 생성된 프로포절 데이터와 폼 입력 데이터를 세션 상태에 저장
                st.session_state['current_proposal_context'] = {
                    "field": field,
                    "topic": topic,
                    "goal": goal,
                    "method": method,
                    "dataset": dataset,
                    "institution": institution,
                    "full_text": result # 생성된 초안 전체 텍스트도 저장
                }

                st.download_button(
                    label="다운로드 (연구 초안)",
                    data=result.encode('utf-8'),
                    file_name="빅데이터_연구_프로포절_초안.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"❌ 프로포절 생성 중 오류가 발생했습니다: {e}")

# --- 챗 메시지 표시 및 채팅 입력 처리 (메인 영역) ---
for msg in st.session_state.messages: # 기존 메시지 표시
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("질문이 있으시면 여기에 입력하세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if api_key:
        try:
            # LLM에 전달할 메시지 리스트 초기화
            chat_messages_for_llm = []

            # 1. 채팅 전용 시스템 프롬프트 (가장 중요: 챗봇의 역할과 목적 명시)
            chat_system_role_prompt = """
            You are a highly skilled and specialized tutor for big data-based research proposals.
            Your main goal is to help the student refine and improve their research proposal draft.
            You should always refer to the *current research proposal context* provided to you below.
            Your responses should be friendly, clear, technically sound, and focused on helping the student revise, expand, clarify, or critique specific sections of their proposal.
            If the student asks a general question, relate it back to their proposal where possible, or provide general guidance as a tutor.
            """
            chat_messages_for_llm.append({"role": "system", "content": chat_system_role_prompt})

            # 2. 이전에 생성된 프로포절 데이터와 폼 입력 정보를 시스템 메시지로 추가 (핵심 맥락 제공)
            if 'current_proposal_context' in st.session_state and st.session_state['current_proposal_context']['full_text']:
                current_proposal_data = st.session_state['current_proposal_context']
                
                # 폼 입력 정보 요약
                form_summary = (
                    f"Student's Form Inputs:\n"
                    f"- Research Field/Subject: {current_proposal_data.get('field', 'N/A')}\n"
                    f"- Research Topic: {current_proposal_data.get('topic', 'N/A')}\n"
                    f"- Research Objective: {current_proposal_data.get('goal', 'N/A')}\n"
                    f"- Analysis Method (Desired): {current_proposal_data.get('method', 'N/A')}\n"
                    f"- Dataset to be Used: {current_proposal_data.get('dataset', 'N/A')}\n"
                    f"- Target Institution/University: {current_proposal_data.get('institution', 'N/A')}\n\n"
                )

                # 전체 프로포절 텍스트
                full_proposal_text = current_proposal_data['full_text']

                # 통합된 프로포절 컨텍스트 시스템 메시지
                chat_messages_for_llm.append({"role": "system", "content": 
                    f"Here is the context of the research proposal you are currently discussing with the student:\n\n"
                    f"{form_summary}"
                    f"--- Generated Proposal Draft ---\n"
                    f"{full_proposal_text}\n"
                    f"-------------------------------\n\n"
                    f"Please use this information as your primary reference for all subsequent interactions."
                })
            else:
                # 프로포절이 아직 생성되지 않았다면 일반적인 튜터 역할 시스템 메시지
                chat_messages_for_llm.append({"role": "system", "content": "You are a helpful tutor assisting students with general questions about big data research proposals. A proposal draft has not been generated yet. Please answer questions based on your general knowledge until a proposal is generated."})

            # 3. 기존 채팅 기록 추가 (현재 대화 세션의 컨텍스트)
            # st.session_state.messages는 이미 화면에 표시되는 메시지 기록입니다.
            # 이 기록을 LLM의 메시지 형식(user, assistant)에 맞게 변환하여 추가합니다.
            for msg_history in st.session_state.messages:
                if msg_history["role"] == "user":
                    chat_messages_for_llm.append({"role": "user", "content": msg_history["content"]})
                elif msg_history["role"] == "tutor":
                    # 튜터의 이전 메시지는 LLM의 이전 응답이므로 'assistant' 역할로 매핑
                    chat_messages_for_llm.append({"role": "assistant", "content": msg_history["content"]})
            
            # 4. 최종 사용자 질문 추가 (if prompt := st.chat_input()에서 이미 prompt 변수에 할당됨)
            # chat_messages_for_llm.append({"role": "user", "content": prompt}) # 이 부분은 위에서 이미 추가된 것으로 간주

            chat_response = client.chat.completions.create(
                model="gpt-4o",
                messages=chat_messages_for_llm, # 구성된 메시지 리스트 사용
                temperature=0.7
            )
            msg = chat_response.choices[0].message.content

        except Exception as e:
            msg = f"채팅 중 오류가 발생했습니다: {e}"
    else:
        msg = "API 키가 설정되지 않아 채팅 기능을 사용할 수 없습니다."

    st.session_state.messages.append({"role": "tutor", "content": msg})
    st.chat_message("tutor").write(msg)
