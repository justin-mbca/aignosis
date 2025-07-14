import gradio as gr
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import json

# ---------------------- 标签映射和模型解释 ----------------------
LABEL_MAPPING = {
    "LABEL_0": {"中文": "低风险", "English": "Low Risk"},
    "LABEL_1": {"中文": "中风险", "English": "Moderate Risk"},
    "LABEL_2": {"中文": "高风险", "English": "High Risk"}
}

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
        "中文": "ClinicalBERT 是针对临床文本优化的模型，适合分析患者相关的临床数据。",
        "English": "ClinicalBERT is optimized for clinical text and is suitable for analyzing patient-related clinical data."
    }
}

# ---------------------- 加载模型 ----------------------
MODELS = {
    "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
    "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT"
}

pipelines = {}
for name, path in MODELS.items():
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=3)
    pipelines[name] = pipeline("text-classification", model=model, tokenizer=tokenizer)

# ---------------------- 心血管疾病分类函数 ----------------------
def classify_cardiovascular_disease(symptoms, history, lab_params, lang="中文"):
    diseases = []
    recommendations = []

    # 高血压判断
    if lang == "中文":
        sbp = lab_params.get("收缩压 (mmHg)", 0)
        dbp = lab_params.get("舒张压 (mmHg)", 0)
        ldl = lab_params.get("低密度脂蛋白 (LDL-C, mg/dL)", 0)
        tc = lab_params.get("总胆固醇 (Total Cholesterol, mg/dL)", 0)
        tg = lab_params.get("甘油三酯 (Triglycerides, mg/dL)", 0)
        troponin = lab_params.get("肌钙蛋白 (Troponin I/T, ng/mL)", 0)
        fasting_glucose = lab_params.get("空腹血糖 (Fasting Glucose, mmol/L)", 0)
        hba1c = lab_params.get("糖化血红蛋白 (HbA1c, %)", 0)
        bmi = lab_params.get("体质指数 (BMI)", 0)
        age = lab_params.get("年龄 (Age)", 0)

        smoking = history.get("是否吸烟？", "否")
        htn_treatment = history.get("是否服用降压药？", "否")
        sex = history.get("性别", "男")

        # 1. 高血压判断
        if sbp > 180 or dbp > 120:
            diseases.append("高血压 (严重)")
            recommendations.append("这是紧急情况，请立即就医。")
        elif sbp > 160 or dbp > 100:
            diseases.append("高血压 (中度)")
            recommendations.append("监测血压，减少盐分摄入，健康饮食，咨询医生。")
        elif sbp > 140 or dbp > 90:
            diseases.append("高血压 (轻度)")
            recommendations.append("定期监测血压，保持健康生活方式。")

        # 2. 冠心病
        if history.get("是否有心脏病家族史？", "否") == "是" or ldl > 130:
            diseases.append("冠心病")
            recommendations.append("建议心脏检查，避免高脂饮食，保持运动。")

        # 3. 心肌梗塞
        if symptoms.get("胸痛是否在劳累时加重？", "否") == "是" and troponin > 0.04:
            diseases.append("心肌梗塞")
            recommendations.append("这是紧急情况，请立即就医。")

        # 4. 高脂血症（更细化）
        if tc > 200 or ldl > 130 or tg > 150:
            diseases.append("高脂血症")
            recommendations.append("低脂饮食，增加纤维素，咨询医生。")

        # 5. 心力衰竭
        if symptoms.get("是否呼吸困难？", "否") == "是" and troponin > 0.1:
            diseases.append("心力衰竭")
            recommendations.append("紧急情况，请立即就医。")

        # 6. 糖尿病
        if history.get("是否患糖尿病？", "否") == "是" or fasting_glucose >= 7.0 or hba1c >= 6.5:
            diseases.append("糖尿病")
            recommendations.append("控制血糖，合理饮食，定期监测。")

        # 7. 肥胖
        if bmi >= 30:
            diseases.append("肥胖")
            recommendations.append("减重，运动，控制饮食。")

        # 8. 心律不齐
        if symptoms.get("是否心悸？", "否") == "是":
            diseases.append("心律不齐")
            recommendations.append("建议心电图检查，排除心律失常。")

        # 9. 心肌炎
        if 0.04 < troponin < 0.1 and symptoms.get("是否呼吸困难？", "否") == "是":
            diseases.append("疑似心肌炎")
            recommendations.append("建议进一步心脏影像学及血清学检查。")

        # 10. Framingham风险评分（极简示范）
        framingham_score = 0
        if sex == "男":
            if age >= 50: framingham_score += 3
            if smoking == "是": framingham_score += 2
            if sbp > 140: framingham_score += 2
            if tc > 200: framingham_score += 2
            if htn_treatment == "是": framingham_score += 1
        else:
            if age >= 50: framingham_score += 2
            if smoking == "是": framingham_score += 2
            if sbp > 140: framingham_score += 1
            if tc > 200: framingham_score += 1
            if htn_treatment == "是": framingham_score += 1

        if framingham_score >= 7:
            diseases.append("Framingham评分高风险")
            recommendations.append("积极控制危险因素，定期复查心血管风险。")
        elif framingham_score >= 4:
            diseases.append("Framingham评分中风险")
            recommendations.append("改善生活方式，监测指标。")
        else:
            diseases.append("Framingham评分低风险")
            recommendations.append("保持健康生活方式。")

        if not diseases:
            diseases.append("无明显心血管疾病风险")
            recommendations.append("保持健康生活方式，定期检查。")

    else:
        # English version logic（可仿照中文逻辑写，省略此处）
        # 请根据实际需求填写
        diseases.append("English version not implemented yet.")
        recommendations.append("Please use Chinese version for now.")

    return diseases, recommendations

# ---------------------- 聚合模型预测 ----------------------
def aggregate_model_predictions(results, lang="中文"):
    if lang == "English":
        risk_labels = ["Low Risk", "Moderate Risk", "High Risk"]
    else:
        risk_labels = ["低风险", "中风险", "高风险"]

    aggregated = {label: 0.0 for label in risk_labels}
    count = 0
    for r in results:
        if "probabilities" in r:
            for risk, score in r["probabilities"].items():
                if risk in aggregated:
                    aggregated[risk] += score
            count += 1
    if count == 0:
        return None, aggregated
    for k in aggregated:
        aggregated[k] /= count
    most_likely = max(aggregated, key=aggregated.get)
    return most_likely, aggregated

# ---------------------- 文本分析和模型调用 ----------------------
def analyze_structured_inputs(symptoms, history, lab_params, lang):
    # 补全默认值
    if lang == "中文":
        symptoms = {k: (v if v is not None else "否") for k, v in symptoms.items()}
        history = {k: (v if v is not None else "否") for k, v in history.items()}
        section_user = "### 📝 用户输入"
        bullet = "🔹"
    else:
        symptoms = {k: (v if v is not None else "No") for k, v in symptoms.items()}
        history = {k: (v if v is not None else "No") for k, v in history.items()}
        section_user = "### 📝 User Inputs"
        bullet = "-"

    # 构造文本用于模型
    text = f"{section_user}:\n\n"
    text += "症状:\n" if lang == "中文" else "Symptoms:\n"
    text += "\n".join([f"{bullet} {k}: {v}" for k, v in symptoms.items()]) + "\n\n"
    text += "病史:\n" if lang == "中文" else "History:\n"
    text += "\n".join([f"{bullet} {k}: {v}" for k, v in history.items()]) + "\n\n"
    text += "实验室参数:\n" if lang == "中文" else "Lab Parameters:\n"
    text += "\n".join([f"{bullet} {k}: {v}" for k, v in lab_params.items()])

    # 调用疾病分类逻辑
    diseases, recommendations = classify_cardiovascular_disease(symptoms, history, lab_params, lang)

    # 调用各模型预测
    model_results = []
    for model_name, clf in pipelines.items():
        try:
            preds = clf(text)
            probabilities = {LABEL_MAPPING[p["label"]][lang]: p["score"] for p in preds}
            most_likely = max(probabilities, key=probabilities.get)
            explanation = MODEL_EXPLANATIONS[model_name][lang]
            model_results.append({
                "model_name": model_name,
                "most_likely": most_likely,
                "probabilities": probabilities,
                "explanation": explanation
            })
        except Exception as e:
            model_results.append({"model_name": model_name, "error": str(e)})

    # 聚合预测
    agg_risk, agg_probs = aggregate_model_predictions(model_results, lang)

    # 结果格式化
    res = [text]
    res.append("### 🩺 疾病分类" if lang == "中文" else "### 🩺 Disease Classification")
    res.append("\n".join([f"{bullet} {d}" for d in diseases]))
    res.append("### 💡 建议" if lang == "中文" else "### 💡 Recommendations")
    res.append("\n".join([f"{bullet} {r}" for r in recommendations]))
    res.append("### 模型预测" if lang == "中文" else "### Model Predictions")
    for r in model_results:
        if "error" in r:
            res.append(f"- {r['model_name']}: Error: {r['error']}")
        else:
            res.append(f"- {r['model_name']}:\n  风险等级: {r['most_likely']}")
            for risk, score in r["probabilities"].items():
                res.append(f"    {risk}: {score:.2f}")
            res.append(f"  模型解释: {r['explanation']}")
    res.append("### 综合风险等级" if lang == "中文" else "### Aggregated Risk Level")
    res.append(f"- 风险等级: {agg_risk}")
    for risk, score in agg_probs.items():
        res.append(f"  {risk}: {score:.2f}")

    return "\n\n".join(res)

# ---------------------- 前端界面 ----------------------
def make_tab(lang):
    yesno = ["是", "否"] if lang == "中文" else ["Yes", "No"]

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
        "近期是否有情绪压力？" if lang == "中文" else "Recent emotional stress?",
        "是否服用降压药？" if lang == "中文" else "Are you on hypertension treatment?",
        "性别" if lang == "中文" else "Sex (Male/Female)"
    ]

    lab_params = [
        ("收缩压 (mmHg)" if lang == "中文" else "Systolic BP (mmHg)", 60, 220, 120),
        ("舒张压 (mmHg)" if lang == "中文" else "Diastolic BP (mmHg)", 40, 120, 80),
        ("低密度脂蛋白 (LDL-C, mg/dL)" if lang == "中文" else "LDL-C (mg/dL)", 50, 200, 100),
        ("高密度脂蛋白 (HDL-C, mg/dL)" if lang == "中文" else "HDL-C (mg/dL)", 20, 100, 50),
        ("总胆固醇 (Total Cholesterol, mg/dL)" if lang == "中文" else "Total Cholesterol (mg/dL)", 100, 300, 200),
        ("甘油三酯 (Triglycerides, mg/dL)" if lang == "中文" else "Triglycerides (mg/dL)", 50, 500, 150),
        ("肌钙蛋白 (Troponin I/T, ng/mL)" if lang == "中文" else "Troponin I/T (ng/mL)", 0.00, 0.50, 0.01),
        ("空腹血糖 (Fasting Glucose, mmol/L)" if lang == "中文" else "Fasting Glucose (mmol/L)", 3.0, 15.0, 5.5),
        ("糖化血红蛋白 (HbA1c, %)" if lang == "中文" else "HbA1c (%)", 3.0, 15.0, 5.0),
        ("体质指数 (BMI)" if lang == "中文" else "BMI", 10, 50, 25),
        ("年龄 (Age)" if lang == "中文" else "Age", 20, 100, 50)
    ]

    # 创建Gradio输入组件
    with gr.Row():
        with gr.Column():
            symptom_inputs = {}
            for q in symptom_questions:
                symptom_inputs[q] = gr.Radio(choices=yesno, label=q, value=yesno[1])
            history_inputs = {}
            for q in history_questions:
                if q == "性别" or q == "Sex (Male/Female)":
                    history_inputs[q] = gr.Radio(choices=["男", "女"] if lang == "中文" else ["Male", "Female"], label=q, value=("男" if lang == "中文" else "Male"))
                else:
                    history_inputs[q] = gr.Radio(choices=yesno, label=q, value=yesno[1])
            lab_inputs = {}
            for label, minimum, maximum, default in lab_params:
                lab_inputs[label] = gr.Slider(minimum, maximum, value=default, label=label)

    def run_model(*args):
        # 解析输入
        n_sym = len(symptom_questions)
        n_hist = len(history_questions)
        n_lab = len(lab_params)
        symptoms = {symptom_questions[i]: args[i] for i in range(n_sym)}
        history = {history_questions[i]: args[n_sym + i] for i in range(n_hist)}
        lab = {lab_params[i][0]: args[n_sym + n_hist + i] for i in range(n_lab)}

        result_text = analyze_structured_inputs(symptoms, history, lab, lang)
        return result_text

    inputs = list(symptom_inputs.values()) + list(history_inputs.values()) + list(lab_inputs.values())
    output = gr.Textbox(label="分析结果", lines=30, interactive=False)

    btn = gr.Button("开始分析" if lang == "中文" else "Analyze")
    btn.click(run_model, inputs=inputs, outputs=output)

    return gr.Column(
        gr.Markdown(f"# 心血管疾病风险评估 ({'中文' if lang == '中文' else 'English'})"),
        *inputs,
        btn,
        output
    )

# ---------------------- 主程序入口 ----------------------
with gr.Blocks() as demo:
    lang_selector = gr.Radio(choices=["中文", "English"], label="选择语言 / Select Language", value="中文")
    output_panel = gr.Column()

    def switch_tab(lang):
        output_panel.children.clear()
        output_panel.children.append(make_tab(lang))

    lang_selector.change(fn=switch_tab, inputs=lang_selector, outputs=output_panel)
    # 初始化默认中文界面
    switch_tab("中文")

demo.launch()
