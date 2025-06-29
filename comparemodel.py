import gradio as gr
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Define label mapping
LABEL_MAPPING = {
    "LABEL_0": "低风险 / Low Risk",
    "LABEL_1": "中风险 / Moderate Risk",
    "LABEL_2": "高风险 / High Risk"
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
    pipelines[model_name] = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Define cardiovascular disease classification logic
def classify_cardiovascular_disease(symptoms, history, lab_params):
    diseases = []
    recommendations = []

    # Hypertension
    if lab_params.get("收缩压 (mmHg)", 0) > 180 or lab_params.get("舒张压 (mmHg)", 0) > 120:
        diseases.append("高血压 (严重) / Hypertension (Severe)")
        recommendations.append("这是紧急情况，请立即就医。")
    elif lab_params.get("收缩压 (mmHg)", 0) > 160 or lab_params.get("舒张压 (mmHg)", 0) > 100:
        diseases.append("高血压 (中度) / Hypertension (Moderate)")
        recommendations.append("建议监测血压，减少盐分摄入，保持健康饮食，并咨询医生。")
    elif lab_params.get("收缩压 (mmHg)", 0) > 140 or lab_params.get("舒张压 (mmHg)", 0) > 90:
        diseases.append("高血压 (轻度) / Hypertension (Mild)")
        recommendations.append("建议定期监测血压，保持健康生活方式。")

    # Coronary Artery Disease (CAD)
    if history.get("是否有心脏病家族史？", "否") == "是" or lab_params.get("低密度脂蛋白 (LDL-C, mg/dL)", 0) > 130:
        diseases.append("冠心病 / Coronary Artery Disease")
        recommendations.append("建议进行心脏健康检查，避免高脂饮食，并保持适度运动。")

    # Myocardial Infarction (MI)
    if symptoms.get("胸痛是否在劳累时加重？", "否") == "是" and lab_params.get("肌钙蛋白 (Troponin I/T, ng/mL)", 0) > 0.04:
        diseases.append("心肌梗塞 / Myocardial Infarction")
        recommendations.append("这是紧急情况，请立即就医。")

    # Hyperlipidemia
    if lab_params.get("总胆固醇 (Total Cholesterol, mg/dL)", 0) > 200 or lab_params.get("低密度脂蛋白 (LDL-C, mg/dL)", 0) > 130:
        diseases.append("高脂血症 / Hyperlipidemia")
        recommendations.append("建议减少高脂饮食，增加富含纤维的食物，并咨询医生。")

    # Heart Failure
    if symptoms.get("是否呼吸困难？", "否") == "是" and lab_params.get("肌钙蛋白 (Troponin I/T, ng/mL)", 0) > 0.1:
        diseases.append("心力衰竭 / Heart Failure")
        recommendations.append("这是紧急情况，请立即就医。")

    # If no diseases are detected
    if not diseases:
        diseases.append("无明显心血管疾病风险 / No significant cardiovascular disease risk detected")
        recommendations.append("保持健康的生活方式，定期进行健康检查。")

    return diseases, recommendations

MODEL_EXPLANATIONS = {
    "BioBERT": "BioBERT 是一个专门针对生物医学文本训练的模型，适用于分析医学相关的文本。",
    "PubMedBERT": "PubMedBERT 是基于 PubMed 数据训练的模型，专注于生物医学文献的理解。",
    "ClinicalBERT": "ClinicalBERT 是针对临床文本（如电子病历）优化的模型，适合分析患者相关的临床数据。"
}

def aggregate_model_predictions(results):
    """
    Aggregates probabilities from all models to determine the overall risk level.
    """
    aggregated_probabilities = {"低风险 / Low Risk": 0, "中风险 / Moderate Risk": 0, "高风险 / High Risk": 0}
    model_count = 0

    for model_result in results:
        if isinstance(model_result, dict):  # Ensure valid results
            for risk, score in model_result["probabilities"].items():
                if risk in aggregated_probabilities:  # Ensure the key exists
                    aggregated_probabilities[risk] += score
            model_count += 1

    # Average the probabilities
    for risk in aggregated_probabilities:
        aggregated_probabilities[risk] /= model_count

    # Determine the most likely risk level
    most_likely = max(aggregated_probabilities, key=aggregated_probabilities.get)
    return most_likely, aggregated_probabilities


def analyze_structured_inputs(symptoms, history, lab_params, lang):
    # Replace None values in symptoms with "否" (No)
    symptoms = {key: (value if value is not None else "否") for key, value in symptoms.items()}

    # Combine structured inputs into a single text representation
    structured_text = (
        f"### 📝 用户输入 / User Inputs:\n\n"
        f"#### 🩺 症状 / Symptoms:\n" +
        "\n".join([f"🔹 {q}: {a}" for q, a in symptoms.items()]) +
        f"\n\n#### 🏥 病史 / Medical History:\n" +
        "\n".join([f"🔹 {q}: {a}" for q, a in history.items()]) +
        f"\n\n#### 🧪 实验室参数 / Lab Parameters:\n" +
        "\n".join([f"🔹 {q}: {a}" for q, a in lab_params.items()])
    )

    # Classify cardiovascular diseases and get recommendations
    diseases, recommendations = classify_cardiovascular_disease(symptoms, history, lab_params)

    # Get predictions from all models
    model_results = []
    for model_name, classifier in pipelines.items():
        try:
            predictions = classifier(structured_text)
            probabilities = {LABEL_MAPPING[pred["label"]]: pred["score"] for pred in predictions}
            most_likely = max(probabilities, key=probabilities.get)
            explanation = MODEL_EXPLANATIONS[model_name]
            model_results.append({
                "model_name": model_name,
                "most_likely": most_likely,
                "probabilities": probabilities,
                "explanation": explanation
            })
        except Exception as e:
            model_results.append({
                "model_name": model_name,
                "error": str(e)
            })

    # Aggregate predictions
    aggregated_risk, aggregated_probabilities = aggregate_model_predictions(model_results)

    # Format the results for display
    formatted_results = [structured_text]  # Include the input string
    formatted_results.append(
        f"### 🩺 疾病分类 / Disease Classification:\n" +
        "\n".join([f"🔹 {disease}" for disease in diseases])
    )
    formatted_results.append(
        f"### 💡 建议 / Recommendations:\n" +
        "\n".join([f"🔹 {recommendation}" for recommendation in recommendations])
    )

    # Add model-specific results
    formatted_results.append("### 模型预测 / Model Predictions:")
    for result in model_results:
        if "error" in result:
            formatted_results.append(f"- **{result['model_name']}**: Error: {result['error']}")
        else:
            formatted_results.append(
                f"- **{result['model_name']}**:\n"
                f"  **风险等级 / Risk Level:** {result['most_likely']}\n"
                f"  **概率分布 / Probability Distribution:**\n" +
                "\n".join([f"    🔹 {risk}: {score:.2f}" for risk, score in result["probabilities"].items()]) +
                f"\n  **模型解释 / Model Explanation:** {result['explanation']}\n"
            )

    # Add aggregated analysis
    formatted_results.append(
        f"### 综合分析 / Aggregated Analysis:\n"
        f"- **综合风险等级 / Overall Risk Level:** {aggregated_risk}\n"
        f"- **综合概率分布 / Aggregated Probability Distribution:**\n" +
        "\n".join([f"  🔹 {risk}: {score:.2f}" for risk, score in aggregated_probabilities.items()]) +
        "\n- **说明 / Explanation:** "
        "模型预测可能存在差异，因为它们基于不同的数据集进行训练。建议根据综合分析结果采取行动，并在必要时咨询医生。"
    )

    return "\n\n".join(formatted_results)

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
            ("低密度脂蛋白 (LDL-C, mg/dL)" if lang == "中文" else "LDL-C (mg/dL)", 50, 200, 100),
            ("高密度脂蛋白 (HDL-C, mg/dL)" if lang == "中文" else "HDL-C (mg/dL)", 20, 100, 50),
            ("总胆固醇 (Total Cholesterol, mg/dL)" if lang == "中文" else "Total Cholesterol (mg/dL)", 100, 300, 200),
            ("肌钙蛋白 (Troponin I/T, ng/mL)" if lang == "中文" else "Troponin I/T (ng/mL)", 0, 50, 0.01)
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
        symptom_fields = [gr.Radio(choices=yesno, label=q) for q in symptom_questions]

    with gr.Group():
        gr.Markdown("### 🏥 病史 / Medical History" if lang == "中文" else "### 🏥 Medical History")
        history_fields = [gr.Radio(choices=yesno, label=q) for q in history_questions]

    with gr.Group():
        gr.Markdown("### 🧪 实验室参数 / Lab Parameters" if lang == "中文" else "### 🧪 Lab Parameters")
        lab_fields = [
            gr.Number(label=q, minimum=minv, maximum=maxv, value=val)
            for q, minv, maxv, val in L["nums"]
        ]

    # Combine all fields
    fields = symptom_fields + history_fields + lab_fields

    # Output and submit button
    output_text = gr.Textbox(label="结果 / Results" if lang == "中文" else "Results")
    submit_button = gr.Button("提交 / Submit" if lang == "中文" else "Submit")

    # Submit button functionality
    submit_button.click(
        fn=lambda *inputs: analyze_structured_inputs(
            symptoms={q: inputs[i] for i, q in enumerate(symptom_questions)},
            history={q: inputs[i + len(symptom_questions)] for i, q in enumerate(history_questions)},
            lab_params={lab_fields[i].label: inputs[i + len(symptom_questions) + len(history_questions)] for i in range(len(lab_fields))},
            lang=lang
        ),
        inputs=fields,
        outputs=[output_text]
    )

    # Create Gradio interface
    return gr.Column(fields + [submit_button, output_text])

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