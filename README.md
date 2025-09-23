# ⭐⭐ Eugenie's Health Data Analysis Portfolio 

This repository showcases practical applications of data science in global health, ranging from LLM-based drug discovery to interactive WHO dashboards. These projects were developed both independently and collaboratively through a data science bootcamp program. Please note that using [GEMINI API](https://ai.google.dev/gemini-api/docs/api-key?hl=ko) provides accessible LLM protocols within free usage limits for experimentation.

I see data not merely as numbers, but as the 'language' of evidence-based policy-making. Building on my background in health economics and global health, investing some time after back home from Zanzibar to upskilling myself, I have been actively developing my data analysis portfolio to design evidence-based policies and serve as a bridge between technology and policy, which is the core value in health financing.

Each project below includes policy implications, highlighting how data-driven insights can inform health and development strategies that leverage both my domain expertise and emerging technical skills. Any comments are welcome 😊


## 🚀 0. LLM Drug Discovery
- Overview: Collaborative project exploring how Large Language Models (LLMs) can accelerate early-stage drug discovery processes.
- Objective: While the team focused on coding implementation, I developed evaluation mechanisms for LLMs in Structure–Activity Relationship (SAR) tasks and proposed systematic evaluation frameworks.
- Dataset(s): Proprietary data provided by a drug discovery startup for this capstone project.
- Methods/Tools: Python (RDKit, Scikit-learn), vector databases, evaluation metrics (including Activity Cliff detection).
- **Outputs**:
    - [GitHub Repository (SARang-Labs)](https://github.com/SARang-Labs/sar-project)
    - [Evaluation Mechanism – Blog Post](https://eugenie-kim012.tistory.com/15)
    - [Notion Portfolio](https://www.notion.so/Capstone-Automating-SAR-Reports-Generation-259bdaab6ba480999b40f9f71e6af975?pvs=21)
- **Policy Implications**: Novel AI evaluation methods for drug discovery can significantly reduce R&D costs and accelerate access to medicines. This has direct relevance for essential, rare and neglected diseases and global health systems facing limited pharmaceutical innovation. Drawing from my graduate research on public finance in R&D and the quantified impact of advance market commitments for vaccination programs in least developed countries, I recognise how market failures in drug discovery for essential technologies can be addressed through innovative approaches like LLM-assisted discovery.


## 📊 1.Mental Health ML Prediction (Datathon)
- **Overview**: As part of the ModuLabs data scientist programme’s Datathon, we collaboratively explored a Kaggle dataset. After generating correlation heatmaps between indicators, I focused on work/study stress, life satisfaction, and the relationship between suicidal thoughts and depression diagnosis to explore increase the accuracy of the algorithm for depression and mental health conditions. Other team members had been explored other components such as the relationships between green space and the mental health, which was still very interesting and relevant.
- **Objective**: The main objective of this Kaggle project was to predict depression in patients using machine learning algorithms, while gaining insights into mental health data patterns.
- **Dataset(s)**: [Kaggle Dataset](https://www.kaggle.com/competitions/playground-series-s4e11)
- **Outputs**:
  - URL: https://github.com/eugenie-kim012/Health_Data/blob/main/1_Mental_health_MLPrediction(Datathon).ipynb
- **Methods/Tools**:  Python (Pandas, Scikit-learn, Matplotlib/Seaborn for visualization)
- **Policy Implications**: Early prediction models for mental health conditions can enable preventive interventions and resource allocation in healthcare systems. Understanding correlations between lifestyle factors (work stress, life satisfaction) and mental health outcomes can inform workplace wellness policies and public health prevention strategies. This approach supports the development of digital health screening tools that could be integrated into primary healthcare settings, particularly valuable in resource-constrained environments where mental health specialists are limited.

## 📊 2. PhD Proposal Development Tutor

- **Overview**:  Developing an LLM-powered tool to support PhD proposal writing in health economics and global health.
- **Objective**: Supporting researchers in developing competitive proposals using AI assistance tailored to health economics and global health domains, particularly for brainstorming and initial conceptualisation.
- **Methods/Tools**: LLM integration, Streamlit, Python
- **Outputs**:
    - URL: https://github.com/eugenie-kim012/Health_Data/blob/main/2_Research_Proposal_Tutor_V1.py
    - [Blog Post](https://eugenie-kim012.tistory.com/7) 
- **Policy Implications**: LLM can be used for the brainstormings!  However, Future development may incorporate evaluation matrices developed through the SAR-Lang project experience.


## 🤖 3. Machine Learning App for Health Data

- **Overview**: Built an end-to-end ML pipeline for health data analysis with user-friendly deployment.
- **Objective**: Deploy a Streamlit application to make machine learning accessible for non-technical users in health policy and healthcare management.
- **Methods/Tools**: Scikit-learn (classification & regression models), Streamlit, Python
- **Outputs**:
    - URL: https://github.com/eugenie-kim012/Health_Data/tree/main/3_MLAPP
    - [Notion Portfolio](https://www.notion.so/Building-a-Machine-Learning-App-for-Health-Data-Analysis-1eebdaab6ba480ffbfd3ef827eb0848a?pvs=21)
    - [Blog Post](https://eugenie-kim012.tistory.com/11)
- **Policy Implications**:
By lowering barriers to ML adoption, such tools empower policy analysts and healthcare managers to test predictive models without coding expertise. This contributes to capacity building in digital health and health systems strengthening, while expanding my analytical portfolio in health economics applications.


## 📑 4. Wise-Aging bot

- **Overview**: An AI chatbot designed to support health and welfare policy development for super-aged societies through enhanced information accessibility and report writing assistance.
- **Objective**:  Provide policymakers with quick reference tools for developing evidence-based policies in aging societies.
- **Dataset(s)**:
    - OECD health and aging-related publications
- **Methods/Tools**: LLM integration, Python, LangChain, RAG (Retrieval-Augmented Generation)
- **Outputs**:
    - URL: https://github.com/eugenie-kim012/Health_Data/tree/main/4_Wise-Aging-Bot
    - [Blog Post](https://eugenie-kim012.tistory.com/8) 
- **Policy Implications**:
This Streamlit-based chatbot leverages LangChain and RAG to explore relevant OECD reports, assist in policy analysis drafting, and provide stakeholder Q&A support. Users can upload PDF reports, filter materials by topic/country/year, and perform policy analysis using the OpenAI API. The system adopts a senior health policy analyst persona to respond to inquiries from policymakers, academics, administrators, and civil society organizations. Source materials were obtained from the [OECD Data Platform](https://www.oecd.org/en/data.html).


## 🌍 5. WHO Data Analysis: Triple Billion Dashboard

- **Overview**: Summer research project leveraging **WHO OPEN Data** to create policy-relevant visualisations.
- **Objective**: Build an interactive dashboard to track WHO's Triple Billion Targets (Universal Health Coverage, health emergency preparedness, and healthier populations).
- **Dataset(s)**:
    - WHO OPEN Data Platform (Triple Billion indicators)
    - Regional health contribution datasets (2018–2023)
- **Methods/Tools**: Streamlit, Plotly, data wrangling with Pandas
- **Impact**: Dashboard visualizes regional contributions and progress toward WHO goals, providing actionable policy insights for global health monitoring and accountability.
- **Outputs**:
    - URL: https://github.com/eugenie-kim012/Health_Data/tree/main/5_Triplebillions
    - [Notion Portfolio](https://www.notion.so/Summer-Break-Data-Analysis-WHO-OPEN-Data-246bdaab6ba480208b37d9b97d8e1390?pvs=21)
    - [Blot Post](https://eugenie-kim012.tistory.com/10)
- **Policy Implications**:
Interactive dashboards enable governments and development partners to monitor progress toward SDGs and WHO targets. This facilitates transparent accountability, evidence-based resource allocation, and informed global health diplomacy decisions.
