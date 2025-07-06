import gradio as gr
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

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
    pipelines[model_name] = pipeline("text-classification", model=model, tokenizer=tokenizer)

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


def analyze_structured_inputs(symptoms, history, lab_params, lang):
    # Replace None values in symptoms with "否"/"No"
    if lang == "中文":
        symptoms = {key: (value if value is not None else "否")
                    for key, value in symptoms.items()}
        section_user = "### 📝 用户输入"
        section_symptoms = "#### 🩺 症状"
        section_history = "#### 🏥 病史"
        section_lab = "#### 🧪 实验室参数"
        section_disease = "### 🩺 疾病分类"
        section_recommend = "### 💡 建议"
        section_model = "### 模型预测"
        section_agg = "### 综合分析"
        bullet = "🔹"
        risk_label = "风险等级"
        prob_label = "概率分布"
        explain_label = "模型解释"
        agg_risk = "综合风险等级"
        agg_prob = "综合概率分布"
        agg_explain = "模型预测可能存在差异，因为它们基于不同的数据集进行训练。建议根据综合分析结果采取行动，并在必要时咨询医生。"
        history = {k: (v if v is not None else "否")
                   for k, v in history.items()}
    else:
        symptoms = {key: (value if value is not None else "No")
                    for key, value in symptoms.items()}
        section_user = "### 📝 User Inputs"
        section_symptoms = "#### 🩺 Symptoms"
        section_history = "#### 🏥 Medical History"
        section_lab = "#### 🧪 Lab Parameters"
        section_disease = "### 🩺 Disease Classification"
        section_recommend = "### 💡 Recommendations"
        section_model = "### Model Predictions"
        section_agg = "### Aggregated Analysis"
        bullet = "🔹"
        risk_label = "Risk Level"
        prob_label = "Probability Distribution"
        explain_label = "Model Explanation"
        agg_risk = "Overall Risk Level"
        agg_prob = "Aggregated Probability Distribution"
        agg_explain = "Model predictions may differ because they are trained on different datasets. Please act according to the combined analysis and consult a doctor if necessary."

    # Combine structured inputs into a single text representation
    structured_text = (
        f"{section_user}:\n\n"
        f"{section_symptoms}:\n" +
        "\n".join([f"{bullet} {q}: {a}" for q, a in symptoms.items()]) +
        f"\n\n{section_history}:\n" +
        "\n".join([f"{bullet} {q}: {a}" for q, a in history.items()]) +
        f"\n\n{section_lab}:\n" +
        "\n".join([f"{bullet} {q}: {a}" for q, a in lab_params.items()])
    )

    # Classify cardiovascular diseases and get recommendations
    diseases, recommendations = classify_cardiovascular_disease(
        symptoms, history, lab_params, lang)

    # Get predictions from all models
    model_results = []
    for model_name, classifier in pipelines.items():
        try:
            predictions = classifier(structured_text)
            print(predictions)
            probabilities = {
                LABEL_MAPPING[pred["label"]][lang]: pred["score"] for pred in predictions}
            most_likely = max(probabilities, key=probabilities.get)
            explanation = MODEL_EXPLANATIONS[model_name][lang]
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
    aggregated_risk, aggregated_probabilities = aggregate_model_predictions(
        model_results, lang)

    # Format the results for display
    formatted_results = [structured_text]
    formatted_results.append(
        f"{section_disease}:\n" +
        "\n".join([f"{bullet} {disease}" for disease in diseases])
    )
    formatted_results.append(
        f"{section_recommend}:\n" +
        "\n".join(
            [f"{bullet} {recommendation}" for recommendation in recommendations])
    )

    # Add model-specific results
    formatted_results.append(f"{section_model}:")
    for result in model_results:
        if "error" in result:
            formatted_results.append(f"- **{result['model_name']}**: Error: {result['error']}")
        else:
            formatted_results.append(
                f"- **{result['model_name']}**:\n"
                f"  **{risk_label}:** {result['most_likely']}\n"
                f"  **{prob_label}:**\n" +
                "\n".join([f"    {bullet} {risk}: {score:.2f}" for risk, score in result["probabilities"].items()]) +
                f"\n  **{explain_label}:** {result['explanation']}\n"
            )

    # Add aggregated analysis
    formatted_results.append(
        f"{section_agg}:\n"
        f"- **{agg_risk}:** {aggregated_risk}\n"
        f"- **{agg_prob}:**\n" +
        "\n".join([f"  {bullet} {risk}: {score:.2f}" for risk, score in aggregated_probabilities.items()]) +
        f"\n- **{('说明' if lang == '中文' else 'Explanation')}:** {agg_explain}"
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

    # File upload section with its own submit and output
    with gr.Group():
        gr.Markdown("📄 上传文件" if lang ==
                    "中文" else "📄 Upload Document")
        # file_input = gr.File(label="上传文件" if lang == "中文" else "Upload File", file_types=[
        #                      ".txt", ".pdf", ".docx"])
    with gr.Row():
        file_input = gr.File(label="上传文件" if lang == "中文" else "Upload File", file_types=[
            ".txt", ".pdf", ".docx"], scale=2)
    file_output = gr.Textbox(
        label="文件分析结果" if lang == "中文" else "File Analysis Result")
    file_submit = gr.Button("分析文件" if lang == "中文" else "Analyze File")
    file_submit.click(
        # fn=lambda file: process_file(file, lang),  # API calls in prcess_file
        inputs=file_input,
        outputs=file_output
    )


    # Create Gradio interface
    return gr.Column(fields + [submit_button, output_text, gr.Markdown("---"), file_input, file_submit, file_output])


def upload_file_interface(lang="English"):
    if lang == "English":
        label = "Upload File"
        placeholder = "Drag and drop or click to upload your file"
        output_label = "File Output"
    else:
        label = "上传文件"
        placeholder = "拖拽或点击上传您的文件"
        output_label = "文件输出"

    with gr.Blocks() as demo:
        gr.Markdown(
            f"## {label} / {('上传文件' if lang == 'English' else 'Upload File')}")
        file_input = gr.File(label=label, file_types=[
                             ".txt", ".pdf", ".docx"], elem_id="file_upload")
        output = gr.Textbox(label=output_label)
        # API call in process_file function
        # file_input.change(process_file, inputs=[file_input, gr.State(lang)], outputs=output)
        gr.Markdown(f"*{placeholder}*")
    return demo

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