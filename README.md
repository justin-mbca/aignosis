# Bilingual Cardiovascular AI Assistant

## Overview

This project is a bilingual (Chinese-English) AI-powered assistant for early cardiovascular risk screening. It combines structured inputs (questionnaires, vitals, lab values), free-text symptom descriptions, and uploaded lab reports. The system leverages multiple biomedical NLP models, rule-based logic, and clinical risk scores to classify user risk and generate explainable, culturally inclusive feedback. Designed for general users and low-resource settings, it’s deployable via lightweight Python apps and Hugging Face Spaces.

---

## Features

- **Bilingual UI (Chinese/English)**: All inputs and outputs are available in both languages, including dual-language medical terminology display.
- **Structured and Free-Text Input**: Users can enter symptoms, history, lab values, and additional concerns in free text.
- **Model Integration**: Uses three biomedical NLP models (BioBERT, PubMedBERT, ClinicalBERT) for risk classification.
- **Rule-Based Disease Logic**: Explicit rules for hypertension, CAD, MI, hyperlipidemia, and heart failure ensure clinical safety.
- **Model Aggregation**: Weighted aggregation of model outputs for robust, explainable results.
- **Lab Report Parsing**: Extracts and maps lab parameters from uploaded files (mock and real), with logic to reconcile overlapping values.
- **HEART Score Calculation**: Integrates a simplified HEART score for additional risk stratification.
- **Free Text Analysis**: Extracts keywords, flags inconsistencies, and can supplement structured data for risk analysis.
- **Explainable Output**: Structured, bilingual summaries with model explanations, clinical alerts, recommendations, and warnings.
- **ChatGPT/LLM Integration**: LLM-powered report summarization.

---

## Technologies Used

- Python 3.9+
- Gradio (web UI)
- Hugging Face Transformers (BioBERT, PubMedBERT, ClinicalBERT)
- OpenAI API (for LLM summarization)
- Custom rule-based logic
- JSON, Regex, and standard Python libraries

---

## Technologies Used Diagram

```mermaid
graph TD
    A(Project) --> B(Python 3.9+)
    A --> C(Gradio)
    A --> D(Hugging Face Transformers)
    D --> D1(BioBERT)
    D --> D2(PubMedBERT)
    D --> D3(ClinicalBERT)
    A --> E(OpenAI API)
    A --> G(Custom Rule-based Logic)
    A --> H(JSON, Regex, Standard Libraries)
```

---

## Model Explanations

- **BioBERT**: Pre-trained on biomedical text, suitable for analyzing medical-related content.
- **PubMedBERT**: Trained on PubMed data, focuses on understanding biomedical literature.
- **ClinicalBERT**: Optimized for clinical text (e.g., electronic medical records), suitable for analyzing patient-related clinical data.

---

## About the Models

- **BioBERT** ([paper](https://arxiv.org/abs/1901.08746), [Hugging Face](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1))
  - A domain-specific language representation model pre-trained on large-scale biomedical corpora (PubMed abstracts and PMC full-text articles).
  - Especially effective for biomedical text mining tasks such as named entity recognition, relation extraction, and question answering.
  - Used here for classifying cardiovascular risk from medical text.

- **PubMedBERT** ([paper](https://arxiv.org/abs/2007.15779), [Hugging Face](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract))
  - The first BERT model pre-trained from scratch on PubMed abstracts only, without general-domain data.
  - Achieves state-of-the-art results on many biomedical NLP tasks, especially those requiring deep understanding of biomedical literature.
  - Used here for risk classification based on structured and free-text medical input.

- **ClinicalBERT** ([paper](https://arxiv.org/abs/1904.05342), [Hugging Face](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT))
  - A BERT model further pre-trained on clinical notes from the MIMIC-III database, making it highly effective for clinical/EMR text.
  - Excels at tasks involving patient records, clinical notes, and other healthcare documentation.
  - Used here to analyze patient symptoms and history for cardiovascular risk.

---

## Workflow Diagram

```mermaid
flowchart TD
    A(User Input: Structured + Free Text + File Upload) --> B(Preprocessing)
    B --> C1(BioBERT<br>ML Model)
    B --> C2(PubMedBERT<br>ML Model)
    B --> C3(ClinicalBERT<br>ML Model)
    B --> D(Rule-based Logic<br>Explicit Clinical Rules)
    B --> E(HEART Score<br>Clinical Risk Score)
    C1 --> F(Aggregation)
    C2 --> F
    C3 --> F
    D --> F
    E --> F
    F --> G(Explainable, Bilingual Output)
    G --> H(User)
```

**Legend:**
- **ML Model:** Machine learning model for risk prediction (BioBERT, PubMedBERT, ClinicalBERT)
- **Rule-based Logic:** Explicit clinical rules for safety
- **HEART Score:** Standard clinical risk score

**Diagram Notes:**
- **ML Model**: Each of BioBERT, PubMedBERT, ClinicalBERT independently predicts risk.
- **Rule-based Logic**: Applies explicit clinical rules (e.g., emergency BP thresholds).
- **HEART Score**: Standardized clinical scoring for chest pain risk.
- All three approaches run in parallel and are aggregated for a robust, explainable, and safe final risk assessment.

---

## System Workflow

1. **User Input**: User selects language, enters symptoms/history/labs, adds free text, and/or uploads lab report.
2. **Preprocessing**: Inputs are normalized, free text is analyzed for keywords and inconsistencies, and lab files are parsed.
3. **Model & Rule Analysis**:
    - Structured summary is sent to three NLP models for risk prediction.
    - Rule-based logic and HEART score are calculated in parallel.
    - Free text can supplement or override structured data if higher risk is detected.
4. **Aggregation**: Model outputs are aggregated (weighted), and final risk is determined (HEART score can override if high).
5. **Output Generation**: Bilingual, explainable report is generated, including model explanations, clinical alerts, recommendations, and warnings.
6. **LLM Summarization**: Output is summarized by a large language model for clarity and personalization.

---

## Example Output

```
## 🩺 Overall risk
🔹 **Moderate Risk**

## 🚨 Clinical Alerts
- Angina symptoms present, please monitor heart health.

## 📊 Model Probability Distribution
### 🔸 BioBERT
- Moderate Risk: 0.41
- High Risk: 0.34
### 🔸 PubMedBERT
- High Risk: 0.34
- Moderate Risk: 0.41
### 🔸 ClinicalBERT
- Moderate Risk: 0.51

## ❤️ HEART Score: 2 points (Moderate Risk)
## ⚖️ Weighted Risk Scores
- Low Risk: 0.12
- Moderate Risk: 0.44
- High Risk: 0.38

## 🩺 Clinical Recommendations
- It is recommended to consult a doctor soon for further cardiac evaluation.

## 💬 Model Explanation
### BioBERT
BioBERT is a model pre-trained on biomedical text, suitable for analyzing medical-related content.
### PubMedBERT
PubMedBERT is trained on PubMed data and focuses on understanding biomedical literature.
### ClinicalBERT
ClinicalBERT is optimized for clinical text (such as electronic medical records) and is suitable for analyzing patient-related clinical data.

## 📝 Input Summary
... (user input summary) ...
```

---

## Safety & Explainability

- **Inconsistency Warnings**: If free text suggests higher risk than structured input, the system warns the user.
- **Dual-Language Display**: All medical terms and recommendations are shown in both Chinese and English.
- **Rule-Based Overrides**: Emergency findings in rules or free text can override model outputs for safety.

---

## Future Directions

- Integrate real LLM (e.g., GPT-4, Claude) for advanced free text extraction and summarization.
- Expand lab report parsing to more formats and languages.
- Add user feedback and continuous learning loop.

---

## License

MIT

## LLM Summarization

The system supports live LLM (Large Language Model) summarization for generating concise, user-friendly reports. When enabled, model outputs are sent to an LLM (such as OpenAI's GPT-4) via API for real-time summary generation. This replaces the previous mock summary and provides more accurate, context-aware explanations.

- **How it works:**
    - The function `summarize_model_outputs` now calls a real LLM API instead of returning a mock summary.
    - Requires a valid API key and internet connection.
    - Summaries are generated dynamically based on the user's input and model results.

- **Configuration:**
    - Set up your API key in the environment or configuration file as described in the code comments.
    - Ensure the `mock` parameter is set to `False` to enable live summarization.
