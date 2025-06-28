from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import gradio as gr

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

# Analyze structured inputs using all models
def analyze_structured_inputs(symptoms, history, lab_params, lang):
    # Combine structured inputs into a single text representation
    structured_text = (
        f"### Input Questions and Answers:\n\n"
        f"Symptoms:\n" +
        "\n".join([f"{q}: {a}" for q, a in symptoms.items()]) +
        f"\n\nHistory:\n" +
        "\n".join([f"{q}: {a}" for q, a in history.items()]) +
        f"\n\nLab Parameters:\n" +
        "\n".join([f"{q}: {a}" for q, a in lab_params.items()])
    )

    # Get predictions from all models
    results = {}
    for model_name, classifier in pipelines.items():
        try:
            predictions = classifier(structured_text)
            probabilities = {LABEL_MAPPING[pred["label"]]: pred["score"] for pred in predictions}
            results[model_name] = probabilities
        except Exception as e:
            results[model_name] = f"Error: {e}"

    # Format the results for display
    formatted_results = [structured_text]  # Include the input string
    for model_name, probabilities in results.items():
        if isinstance(probabilities, dict):
            formatted_results.append(
                f"### {model_name} Predictions:\n" +
                "\n".join([f"{risk}: {prob:.2f}" for risk, prob in probabilities.items()])
            )
        else:
            formatted_results.append(f"### {model_name} Predictions:\n{probabilities}")

    return "\n\n".join(formatted_results)

# Create Gradio interface for each language
def make_tab(lang):
    L = {
        "yes": "是" if lang == "中文" else "Yes",
        "no": "否" if lang == "中文" else "No",
        "nums": [
            (
                "收缩压 (mmHg)" if lang == "中文" else "Systolic BP (mmHg)",
                60, 220, 120
            ),
            (
                "舒张压 (mmHg)" if lang == "中文" else "Diastolic BP (mmHg)",
                40, 120, 80
            ),
            (
                "低密度脂蛋白 (LDL-C, mg/dL)" if lang == "中文" else "LDL-C (mg/dL)",
                50, 200, 100
            ),
            (
                "高密度脂蛋白 (HDL-C, mg/dL)" if lang == "中文" else "HDL-C (mg/dL)",
                20, 100, 50
            ),
            (
                "总胆固醇 (Total Cholesterol, mg/dL)" if lang == "中文" else "Total Cholesterol (mg/dL)",
                100, 300, 200
            ),
            (
                "肌钙蛋白 (Troponin I/T, ng/mL)" if lang == "中文" else "Troponin I/T (ng/mL)",
                0, 50, 0.01
            )
        ]
    }
    yesno = [L["yes"], L["no"]]
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
        gr.Markdown("### 症状 / Symptoms" if lang == "中文" else "### Symptoms")
        symptom_fields = [gr.Radio(choices=yesno, label=q) for q in symptom_questions]

    with gr.Group():
        gr.Markdown("### 病史 / Medical History" if lang == "中文" else "### Medical History")
        history_fields = [gr.Radio(choices=yesno, label=q) for q in history_questions]

    with gr.Group():
        gr.Markdown("### 实验室参数 / Lab Parameters" if lang == "中文" else "### Lab Parameters")
        lab_fields = [
            gr.Number(label=q, minimum=minv, maximum=maxv, value=val)
            for q, minv, maxv, val in L["nums"]
        ]

    # Combine all fields
    fields = symptom_fields + history_fields + lab_fields

    # Output and submit button
    output = gr.Textbox(label="模型比较结果 / Model Comparisons" if lang == "中文" else "Model Comparisons")
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
        outputs=output
    )

    # Create Gradio interface
    return gr.Column(fields + [submit_button, output])

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