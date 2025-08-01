import gradio as gr
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import openai
import docx
import os
from dotenv import load_dotenv
from process_file import extract_key_value_pairs
import json
import re

# Load environment variables from .env file
load_dotenv()

# Define label mapping
LABEL_MAPPING = {
    "LABEL_0": {
        "中文": "低风险",
        "English": "Low Risk"
    },
    "LABEL_1": {
        "中文": "中风险",
        "English": "Moderate Risk"
    },
    "LABEL_2": {
        "中文": "高风险",
        "English": "High Risk"
    }
}

# Define the models
MODELS = {
    "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
    "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT"
}

# Load pipelines for each model
pipelines = {}
for model_name, model_path in MODELS.items():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
    pipelines[model_name] = pipeline(
        "text-classification", model=model, tokenizer=tokenizer)

# Define cardiovascular disease classification logic


def classify_cardiovascular_disease(symptoms, history, lab_params, lang="中文"):
    diseases = []
    recommendations = []

    if lang == "English":
        # Hypertension
        if lab_params.get("Systolic BP (mmHg)", 0) > 180 or lab_params.get("Diastolic BP (mmHg)", 0) > 120:
            diseases.append("Hypertension (Severe)")
            recommendations.append(
                "This is an emergency. Please seek medical attention immediately.")
        elif lab_params.get("Systolic BP (mmHg)", 0) > 160 or lab_params.get("Diastolic BP (mmHg)", 0) > 100:
            diseases.append("Hypertension (Moderate)")
            recommendations.append(
                "Monitor your blood pressure, reduce salt intake, maintain a healthy diet, and consult your doctor.")
        elif lab_params.get("Systolic BP (mmHg)", 0) > 140 or lab_params.get("Diastolic BP (mmHg)", 0) > 90:
            diseases.append("Hypertension (Mild)")
            recommendations.append(
                "Regularly monitor your blood pressure and maintain a healthy lifestyle.")

        # Coronary Artery Disease (CAD)
        if history.get("Family history of heart disease?", "No") == "Yes" or lab_params.get("LDL-C (mg/dL)", 0) > 130:
            diseases.append("Coronary Artery Disease")
            recommendations.append(
                "Consider a cardiac health check, avoid high-fat diets, and maintain regular exercise.")

        # Myocardial Infarction (MI)
        if symptoms.get("Chest pain triggered by exertion?", "No") == "Yes" and lab_params.get("Troponin I/T (ng/mL)", 0) > 0.04:
            diseases.append("Myocardial Infarction")
            recommendations.append(
                "This is an emergency. Please seek medical attention immediately.")

        # Hyperlipidemia
        if lab_params.get("Total Cholesterol (mg/dL)", 0) > 200 or lab_params.get("LDL-C (mg/dL)", 0) > 130:
            diseases.append("Hyperlipidemia")
            recommendations.append(
                "Reduce high-fat foods, increase fiber-rich foods, and consult your doctor.")

        # Heart Failure
        if symptoms.get("Shortness of breath?", "No") == "Yes" and lab_params.get("Troponin I/T (ng/mL)", 0) > 0.1:
            diseases.append("Heart Failure")
            recommendations.append(
                "This is an emergency. Please seek medical attention immediately.")

        if not diseases:
            diseases.append(
                "No significant cardiovascular disease risk detected")
            recommendations.append(
                "Maintain a healthy lifestyle and have regular health check-ups.")

    else:
        # Hypertension
        if lab_params.get("收缩压 (mmHg)", 0) > 180 or lab_params.get("舒张压 (mmHg)", 0) > 120:
            diseases.append("高血压 (严重)")
            recommendations.append("这是紧急情况，请立即就医。")
        elif lab_params.get("收缩压 (mmHg)", 0) > 160 or lab_params.get("舒张压 (mmHg)", 0) > 100:
            diseases.append("高血压 (中度)")
            recommendations.append("建议监测血压，减少盐分摄入，保持健康饮食，并咨询医生。")
        elif lab_params.get("收缩压 (mmHg)", 0) > 140 or lab_params.get("舒张压 (mmHg)", 0) > 90:
            diseases.append("高血压 (轻度)")
            recommendations.append("建议定期监测血压，保持健康生活方式。")

        # Coronary Artery Disease (CAD)
        if history.get("是否有心脏病家族史？", "否") == "是" or lab_params.get("低密度脂蛋白 (LDL-C, mg/dL)", 0) > 130:
            diseases.append("冠心病")
            recommendations.append("建议进行心脏健康检查，避免高脂饮食，并保持适度运动。")

        # Myocardial Infarction (MI)
        if symptoms.get("胸痛是否在劳累时加重？", "否") == "是" and lab_params.get("肌钙蛋白 (Troponin I/T, ng/mL)", 0) > 0.04:
            diseases.append("心肌梗塞")
            recommendations.append("这是紧急情况，请立即就医。")

        # Hyperlipidemia
        if lab_params.get("总胆固醇 (Total Cholesterol, mg/dL)", 0) > 200 or lab_params.get("低密度脂蛋白 (LDL-C, mg/dL)", 0) > 130:
            diseases.append("高脂血症")
            recommendations.append("建议减少高脂饮食，增加富含纤维的食物，并咨询医生。")

        # Heart Failure
        if symptoms.get("是否呼吸困难？", "否") == "是" and lab_params.get("肌钙蛋白 (Troponin I/T, ng/mL)", 0) > 0.1:
            diseases.append("心力衰竭")
            recommendations.append("这是紧急情况，请立即就医。")

        if not diseases:
            diseases.append("无明显心血管疾病风险")
            recommendations.append("保持健康的生活方式，定期进行健康检查。")

    return diseases, recommendations

MODEL_EXPLANATIONS = {
    "BioBERT": {
        "中文": "BioBERT 是一个专门针对生物医学文本训练的模型，适用于分析医学相关的文本。",
        "English": "BioBERT is a model pre-trained on biomedical text, suitable for analyzing medical-related content."
    },
    "PubMedBERT": {
        "中文": "PubMedBERT 是基于 PubMed 数据训练的模型，专注于生物医学文献的理解。",
        "English": "PubMedBERT is trained on PubMed data and focuses on understanding biomedical literature."
    },
    "ClinicalBERT": {
        "中文": "ClinicalBERT 是针对临床文本（如电子病历）优化的模型，适合分析患者相关的临床数据。",
        "English": "ClinicalBERT is optimized for clinical text (such as electronic medical records) and is suitable for analyzing patient-related clinical data."
    }
}


def aggregate_model_predictions(results, lang="中文"):
    """
    Aggregates probabilities from all models to determine the overall risk level,
    and outputs labels in the specified language.
    """
    # Define risk labels based on language
    if lang == "English":
        risk_labels = ["Low Risk", "Moderate Risk", "High Risk"]
    else:
        risk_labels = ["低风险", "中风险", "高风险"]

    # Initialize aggregated probabilities
    aggregated_probabilities = {label: 0 for label in risk_labels}
    model_count = 0

    for model_result in results:
        if isinstance(model_result, dict) and "probabilities" in model_result:
            for risk, score in model_result["probabilities"].items():
                # Only aggregate if the risk label matches the current language
                if risk in aggregated_probabilities:
                    aggregated_probabilities[risk] += score
            model_count += 1

    # Avoid division by zero
    if model_count == 0:
        return None, aggregated_probabilities

    # Average the probabilities
    for risk in aggregated_probabilities:
        aggregated_probabilities[risk] /= model_count

    # Determine the most likely risk level
    most_likely = max(aggregated_probabilities, key=aggregated_probabilities.get)
    return most_likely, aggregated_probabilities

def generate_summary_text(symptoms, history, lab_params, lang):
    """
    Generate a summary text from structured inputs for model analysis.
    """
    if lang == "中文":
        section_user = "### 📝 用户输入"
        section_symptoms = "#### 🩺 症状"
        section_history = "#### 🏥 病史"
        section_lab = "#### 🧪 实验室参数"
        bullet = "🔹"
    else:
        section_user = "### 📝 User Inputs"
        section_symptoms = "#### 🩺 Symptoms"
        section_history = "#### 🏥 Medical History"
        section_lab = "#### 🧪 Lab Parameters"
        bullet = "🔹"

    summary = (
        f"{section_user}:\n\n"
        f"{section_symptoms}:\n" +
        "\n".join([f"{bullet} {q}: {a}" for q, a in symptoms.items()]) +
        f"\n\n{section_history}:\n" +
        "\n".join([f"{bullet} {q}: {a}" for q, a in history.items()]) +
        f"\n\n{section_lab}:\n" +
        "\n".join([f"{bullet} {q}: {a}" for q, a in lab_params.items()])
    )
    return summary

def generate_recommendations(final_risk, heart_score, lang):
    recommendations = []
    if lang == "中文":
        if final_risk == "高风险":
            recommendations.append("这是紧急情况，请立即就医。")
        elif final_risk == "中风险":
            recommendations.append("建议尽快咨询医生，进一步检查心脏健康。")
        else:
            recommendations.append("风险较低，建议定期体检，保持健康生活方式。")
        if heart_score >= 4:
            recommendations.append(f"HEART评分较高（{heart_score}分），请高度重视心脏健康。")
    else:
        if final_risk == "High Risk":
            recommendations.append("This is an emergency. Please seek medical attention immediately.")
        elif final_risk == "Moderate Risk":
            recommendations.append("It is recommended to consult a doctor soon for further cardiac evaluation.")
        else:
            recommendations.append("Risk is low. Regular check-ups and a healthy lifestyle are recommended.")
        if heart_score >= 4:
            recommendations.append(f"HEART score is high ({heart_score} points). Please pay close attention to your heart health.")
    return recommendations

def generate_clinical_alerts(symptoms, history, lab_params, lang):
    alerts = []
    # Example: Troponin alert
    if lang == "中文":
        if lab_params.get("肌钙蛋白 (Troponin I/T, ng/mL)", 0) > 0.04:
            alerts.append("肌钙蛋白升高，提示心肌损伤风险。")
        if lab_params.get("收缩压 (mmHg)", 0) > 180 or lab_params.get("舒张压 (mmHg)", 0) > 120:
            alerts.append("血压极高，存在高血压急症风险。")
        if symptoms.get("胸痛是否在劳累时加重？", "否") == "是":
            alerts.append("存在心绞痛症状，请注意心脏健康。")
    else:
        if lab_params.get("Troponin I/T (ng/mL)", 0) > 0.04:
            alerts.append("Elevated troponin indicates risk of myocardial injury.")
        if lab_params.get("Systolic BP (mmHg)", 0) > 180 or lab_params.get("Diastolic BP (mmHg)", 0) > 120:
            alerts.append("Extremely high blood pressure, risk of hypertensive emergency.")
        if symptoms.get("Is chest pain aggravated by exertion?", "No") == "Yes":
            alerts.append("Angina symptoms present, please monitor heart health.")
    return alerts

def calculate_heart_score(symptoms, history, lab_params, lang):
    """
    Calculate a simplified HEART score based on inputs.
    Returns (score, risk_level).
    """
    score = 0

    # Example scoring logic (customize for your needs)
    # History
    if history.get("是否有心脏病家族史？", "否") == "是" or history.get("Family history of heart disease?", "No") == "Yes":
        score += 1
    if history.get("是否患有高血压？", "否") == "是" or history.get("Do you have hypertension?", "No") == "Yes":
        score += 1
    if history.get("是否患糖尿病？", "否") == "是" or history.get("Do you have diabetes?", "No") == "Yes":
        score += 1

    # Symptoms
    if symptoms.get("胸痛是否在劳累时加重？", "否") == "是" or symptoms.get("Is chest pain aggravated by exertion?", "No") == "Yes":
        score += 2
    if symptoms.get("是否呼吸困难？", "否") == "是" or symptoms.get("Is there shortness of breath?", "No") == "Yes":
        score += 1

    # Lab parameters (example: Troponin)
    if lab_params.get("肌钙蛋白 (Troponin I/T, ng/mL)", 0) > 0.04 or lab_params.get("Troponin I/T (ng/mL)", 0) > 0.04:
        score += 2

    # Risk level mapping
    if score >= 4:
        risk = "高风险" if lang == "中文" else "High Risk"
    elif score >= 2:
        risk = "中风险" if lang == "中文" else "Moderate Risk"
    else:
        risk = "低风险" if lang == "中文" else "Low Risk"

    return score, risk


def handle_file_output(file_output, lang):
    """
    Process uploaded files，return file_data, file_mapping, file_section
    """
    file_data = None
    file_mapping = None
    file_section = None
    if file_output:
        file_data = process_file(file_output, lang)
        if isinstance(file_data, str):
            try:
                file_data = json.loads(file_data)
            except json.JSONDecodeError:
                return None, None, f"Error processing file: {file_data}"
    if file_data:
        file_section = "### 上传文件内容解析" if lang == "中文" else "### File Content Analysis"
        file_mapping = map_uploaded_file(file_data)
        print(f"File mapping: {file_mapping}")
    return file_data, file_mapping, file_section


def analyze_structured_inputs(symptoms, history, lab_params, file_output, lang):
    # 1. Process uploaded file if provided
    file_data, file_mapping, file_section = handle_file_output(
        file_output, lang)
    if file_data:
        # Merge overlapping lab parameters
        overlap_keys = []
        for k, v in file_mapping.items():
            if k in lab_params:
                if lang == "中文":
                    overlap_keys.append(
                        f"文件覆盖实验室参数： {k}:{v} 替换  {lab_params[k]}")
                else:
                    overlap_keys.append(
                        f"Overriding lab parameter {k}:{v} with original value {lab_params[k]}")
                lab_params[k] = v
    print(f"Processing symptoms: {symptoms}")
    print(f"Processing history: {history}")
    print(f"Processing lab parameters: {lab_params}")
    # 2. Generate summary text
    summary = generate_summary_text(symptoms, history, lab_params, lang)

    # 3. Classify cardiovascular diseases and get recommendations
    diseases, recommendations = classify_cardiovascular_disease(
        symptoms, history, lab_params, lang)
    
    # 4. Model predictions (weighted aggregation)
    model_weights = {"BioBERT": 0.3, "ClinicalBERT": 0.3, "PubMedBERT": 0.4}
    risk_labels = ["低风险", "中风险", "高风险"] if lang == "中文" else ["Low Risk", "Moderate Risk", "High Risk"]
    risk_scores = {label: 0 for label in risk_labels}
    outputs = {}
    for model_name, clf in pipelines.items():
        predictions = clf(summary)
        result = {LABEL_MAPPING[p['label']][lang]: p['score'] for p in predictions if p['label'] in LABEL_MAPPING}
        sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        outputs[model_name] = (sorted_result, result)
        for label, score in result.items():
            if model_name in model_weights and label in risk_scores:
                risk_scores[label] += score * model_weights[model_name]

    ai_risk = max(risk_scores, key=risk_scores.get)

    # 5. HEART score
    heart_score, heart_risk = calculate_heart_score(symptoms, history, lab_params, lang)

    # 6. Final risk level
    final_risk = heart_risk if heart_score >= 4 else ai_risk

    # 7. Clinical alerts
    alerts = generate_clinical_alerts(symptoms, history, lab_params, lang)

    # 8. Recommendations
    recommendations = generate_recommendations(final_risk, heart_score, lang)

    # 9. Output formatting
    output = f"## 🩺 综合风险等级\n🔹 **{final_risk}**\n\n" if lang == "中文" else f"## 🩺 Overall risk\n🔹 **{final_risk}**\n\n"
    if alerts:
        output += "## 🚨 临床警报\n" if lang == "中文" else "## 🚨 Clinical Alerts\n"
        for alert in alerts:
            output += f"- {alert}\n"
        output += "\n"
    output += "## 📊 模型概率分布\n" if lang == "中文" else "## 📊 Model Probability Distribution\n"
    for model_name in outputs:
        output += f"### 🔸 {model_name}\n"
        for label, score in outputs[model_name][0]:
            output += f"- {label}: {score:.2f}\n"
    output += f"\n## ❤️ HEART评分: {heart_score}分 ({heart_risk})\n" if lang == "中文" else f"\n## ❤️ HEART Score: {heart_score} points ({heart_risk})\n"
    output += "## ⚖️ 加权风险分数\n" if lang == "中文" else "## ⚖️ Weighted Risk Scores\n"
    for risk, score in risk_scores.items():
        output += f"- {risk}: {score:.3f}\n"
    output += "\n## 🩺 临床建议\n" if lang == "中文" else "\n## 🩺 Clinical Recommendations\n"
    for rec in recommendations:
        output += f"- {rec}\n"
    output += f"\n## 💬 模型说明\n" if lang == "中文" else f"\n## 💬 Model Explanation\n"
    for model_name in outputs:
        output += f"### {model_name}\n"
        output += f"{MODEL_EXPLANATIONS.get(model_name, {}).get(lang, '暂无说明' if lang == '中文' else 'No description available')}\n\n"
    output += f"\n## 📝 输入摘要\n{summary}\n" if lang == "中文" else f"\n## 📝 Input Summary\n{summary}\n"
    if file_data:
        output += f"{file_section}"
        output += f"\n{json.dumps(file_data, indent=2, ensure_ascii=False)}\n\n"
    if overlap_keys:
        output += "\n".join(overlap_keys)
    return output

# Create Gradio interface for each language
def make_tab(lang):
    """
    Creates a tab for the specified language (Chinese or English).
    """
    L = {
        "yes": "是" if lang == "中文" else "Yes",
        "no": "否" if lang == "中文" else "No",
        "nums": [
            ("收缩压 (mmHg)" if lang == "中文" else "Systolic BP (mmHg)", 60, 220, 120),
            ("舒张压 (mmHg)" if lang == "中文" else "Diastolic BP (mmHg)", 40, 120, 80),
            ("低密度脂蛋白胆固醇 (mg/dL)" if lang ==
             "中文" else "LDL Cholesterol (mg/dL)", 50, 200, 100),
            ("高密度脂蛋白胆固醇 (mg/dL)" if lang ==
             "中文" else "HDL Cholesterol (mg/dL)", 20, 100, 50),
            ("总胆固醇 (mg/dL)" if lang ==
             "中文" else "Total Cholesterol (mg/dL)", 0, 300, 200),
            ("肌钙蛋白 (Troponin I/T, ng/mL)" if lang ==
             "中文" else "Troponin I/T (ng/mL)", 0, 50, 0.01)
        ]
    }
    yesno = [L["yes"], L["no"]]

    # Grouped questions
    symptom_questions = [
        "胸痛是否在劳累时加重？" if lang == "中文" else "Is chest pain aggravated by exertion?",
        "是否为压迫感或紧缩感？" if lang == "中文" else "Is it a pressing or tightening sensation?",
        "是否持续超过5分钟？" if lang == "中文" else "Does it last more than 5 minutes?",
        "是否放射至肩/背/下巴？" if lang == "中文" else "Does it radiate to shoulder/back/jaw?",
        "是否在休息后缓解？" if lang == "中文" else "Is it relieved by rest?",
        "是否伴冷汗？" if lang == "中文" else "Is it accompanied by cold sweat?",
        "是否呼吸困难？" if lang == "中文" else "Is there shortness of breath?",
        "是否恶心或呕吐？" if lang == "中文" else "Is there nausea or vomiting?",
        "是否头晕或晕厥？" if lang == "中文" else "Is there dizziness or fainting?",
        "是否心悸？" if lang == "中文" else "Is there palpitations?"
    ]
    history_questions = [
        "是否患有高血压？" if lang == "中文" else "Do you have hypertension?",
        "是否患糖尿病？" if lang == "中文" else "Do you have diabetes?",
        "是否有高血脂？" if lang == "中文" else "Do you have hyperlipidemia?",
        "是否吸烟？" if lang == "中文" else "Do you smoke?",
        "是否有心脏病家族史？" if lang == "中文" else "Family history of heart disease?",
        "近期是否有情绪压力？" if lang == "中文" else "Recent emotional stress?"
    ]

    # Create Gradio components

    with gr.Group():
        gr.Markdown("### 🩺 症状 / Symptoms" if lang == "中文" else "### 🩺 Symptoms")
        symptom_fields = [gr.Radio(choices=yesno, label=q, value=None)
                          for q in symptom_questions]

    with gr.Group():
        gr.Markdown("### 🏥 病史 / Medical History" if lang == "中文" else "### 🏥 Medical History")
        history_fields = [gr.Radio(choices=yesno, label=q, value=None)
                          for q in history_questions]

    with gr.Group():
        gr.Markdown("### 🧪 实验室参数 / Lab Parameters" if lang == "中文" else "### 🧪 Lab Parameters")
        lab_fields = [
            gr.Number(label=q, minimum=minv, maximum=maxv, value=val)
            for q, minv, maxv, val in L["nums"]
        ]


    with gr.Group():
        label = "上传文件" if lang == "中文" else "Upload File"
        gr.Markdown(label)
        file_input = gr.File(label=label, file_types=[
                             ".txt", ".pdf", ".docx"], elem_id="file_upload")

    # Combine all fields
    fields = symptom_fields + history_fields + lab_fields + [file_input]

    # Output and submit button
    output_text = gr.Textbox(label="结果 / Results" if lang == "中文" else "Results")
    reset_button = gr.Button("重置" if lang == "中文" else "Reset")
    submit_button = gr.Button("提交 / Submit" if lang == "中文" else "Submit")

    # Submit button functionality
    submit_button.click(
        fn=lambda *inputs: analyze_structured_inputs(
            symptoms={q: inputs[i] for i, q in enumerate(symptom_questions)},
            history={q: inputs[i + len(symptom_questions)] for i, q in enumerate(history_questions)},
            lab_params={
                lab_fields[i].label: inputs[i +
                                            len(symptom_questions) + len(history_questions)]
                for i in range(len(lab_fields))
                if inputs[i + len(symptom_questions) + len(history_questions)] not in (None, 0)
            },
            file_output=inputs[-1],
            lang=lang
        ),
        inputs=fields,
        outputs=[output_text]
    )

    default_values = (
        [None] * len(symptom_fields) +
        [None] * len(history_fields) +
        [val for q, minv, maxv, val in L["nums"]] +
        [None]  # file_input
    )

    reset_button.click(
        fn=lambda: default_values,
        inputs=None,
        outputs=fields
    )

    # Create Gradio interface
    return gr.Column(fields + [submit_button, output_text, reset_button, gr.Markdown("---")])

def process_file(file, lang="English", mock=True):
    """
    Process the uploaded docx file and use OpenAI API to extract key-value pairs.
    If mock is True, return a fixed JSON structure for testing.
    """
    if mock:
        if lang == "中文":
            mock_data = {
                "癌胚抗原 (CEA)": "3.22 ng/ml (≤5)",
                "甲胎蛋白": "3.52 ng/ml (≤7)",
                "高密度脂蛋白胆固醇": "78.15 mg/dL (>40)",
                "低密度脂蛋白胆固醇": "171.4 mg/dL (<130) ↑",
                "甘油三酯": "110.7 mg/dL (<150)",
                "总胆固醇": "266.5 mg/dL (<200) ↑",
                "尿素": "37.64 mg/dL (18.63–52.85)",
                "总二氧化碳": "26.8 mEq/L (22.0–29.0)",
                "尿酸": "3.97 mg/dL (2.61–6.00)",
                "肌酐": "0.71 mg/dL (0.46–0.92)"
            }
        else:
            mock_data = {
                "LDL Cholesterol": "84 mg/dL (Ref: < 135 mg/dL)",
                "Total Cholesterol": "185 mg/dL (Ref: < 200 mg/dL)",
                "HDL Cholesterol": "76 mg/dL (Ref: ≥ 40 mg/dL)",
                "Non-HDL Cholesterol": "109 mg/dL (Ref: < 162 mg/dL)",
                "Triglycerides": "144 mg/dL (Ref: < 150 mg/dL)",
                "A1c": "5.4% (Ref: < 6.0%)",
                "eGFR": "72 mL/min/1.73m² (Ref: ≥ 60)",
                "Urea (BUN equivalent)": "23 mg/dL (Ref: ~7 – 23 mg/dL)",
                "Iron": "67 µg/dL (Ref: 40 – 160 µg/dL)",
                "Vitamin B12": "149 pg/mL (Ref: 148–220: Insufficiency)",
                "PSA (Prostate Specific Antigen)": "1.68 ng/mL (Ref: < 3.5 ng/mL)",
                "DHEAS": "148 µg/dL (Ref: ~69 – 305 µg/dL)",
                "WBC Count": "4.1 x10⁹/L (Ref: 4.5 – 11.0 x10⁹/L)",
                "RBC Count": "4.9 x10¹²/L (Ref: 4.4 – 5.9 x10¹²/L)",
                "Hemoglobin": "15.2 g/dL (Ref: 14.0 – 18.0 g/dL)",
                "Lymphocytes": "0.8 x10⁹/L (Ref: 1.0 – 3.3 x10⁹/L)",
                "Ferritin": "473 ng/mL (Ref: > 220 ng/mL)",
                "Platelets": "170 x10⁹/L (Ref: 140 – 440 x10⁹/L)"
            }
        
        return json.dumps(mock_data, indent=2, ensure_ascii=False)
    if file is None:
        return "No file uploaded."
    try:
        # Save uploaded file to a temp path
        temp_path = file.name
        result = extract_key_value_pairs(temp_path)
        if result is None:
            return "Could not extract key-value pairs. See logs for details."
        # Pretty print JSON result
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error processing file: {e}"


def map_uploaded_file(data):
    """
    Map the uploaded file to the appropriate key-value pairs.
    支持 value 为 "数值 单位 (参考范围)" 的字符串格式。
    """
    if data is None:
        return "No content returned."
    result = {}
    for name, entry in data.items():
        # 用正则提取数值和单位
        match = re.match(r"([-\d.]+)\s*([a-zA-Zµ/%]+)", entry)
        if match:
            value_str, unit = match.groups()
            try:
                value = float(value_str)
                result[f"{name} ({unit})"] = value
            except Exception:
                continue
    return result


# Launch Gradio app
if __name__ == "__main__":
    with gr.Blocks() as app:
        gr.Markdown("## 🌐 智能心血管评估系统 | Bilingual Cardiovascular Assistant")
        with gr.Tabs():
            with gr.TabItem("中文"):
                make_tab("中文")
            with gr.TabItem("English"):
                make_tab("English")
        app.launch(share=True)