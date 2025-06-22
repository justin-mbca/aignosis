import gradio as gr
from transformers import pipeline

# Load the Hugging Face pipeline for text classification
text_analysis_pipeline = pipeline("text-classification",
                                  model="dmis-lab/biobert-base-cased-v1.1",
                                  from_pt=True)  # Force loading the model using PyTorch weights)

# 定义标签映射
LABEL_MAPPING = {
    "LABEL_0": "低风险 / Low Risk",
    "LABEL_1": "中风险 / Moderate Risk",
    "LABEL_2": "高风险 / High Risk"
}

# 使用 Hugging Face 模型分析自由文本
def analyze_free_text(free_text):
    if not free_text.strip():
        return "无额外信息 / No additional information provided."
    
    try:
        results = text_analysis_pipeline(free_text)
        analysis = "\n".join([
            f"{LABEL_MAPPING.get(label['label'], label['label'])}: {label['score']:.2f}"
            for label in results
        ])
        print(f"Debug: Free Text Analysis = {analysis}")  # 调试输出
        return f"分析结果 / Analysis Results:\n{analysis}"
    except Exception as e:
        print(f"Error in analyze_free_text: {e}")  # 错误日志
        return f"无法分析自由文本信息 / Unable to analyze free text information: {e}"

# 检测结构化问题和自由文本分析的冲突
def detect_conflicts(structured_result, huggingface_analysis):
    if "低风险" in structured_result and "高风险" in huggingface_analysis:
        return True
    if "高风险" in structured_result and "低风险" in huggingface_analysis:
        return True
    return False

# 评估心血管疾病类型
def evaluate_cardiovascular_disease(symptoms, history, lab_params):
    diseases = []

    print(f"Debug: Symptoms = {symptoms}")
    print(f"Debug: History = {history}")
    print(f"Debug: Lab Parameters = {lab_params}")

    # 高血压（Hypertension）
    if lab_params.get("Systolic BP", 0) > 140 or lab_params.get("Diastolic BP", 0) > 90:
        diseases.append("高血压 / Hypertension")
        print("Debug: Detected 高血压 / Hypertension")

    # 冠心病（Coronary Artery Disease, CAD）
    if history.get("Family History of Heart Disease", False) or lab_params.get("LDL-C", 0) > 130:
        diseases.append("冠心病 / Coronary Artery Disease")
        print("Debug: Detected 冠心病 / Coronary Artery Disease")

    # 心肌梗塞（Myocardial Infarction, MI）
    if symptoms.get("Chest Pain", False) and lab_params.get("Troponin I/T", 0) > 0.04:
        diseases.append("心肌梗塞 / Myocardial Infarction")
        print("Debug: Detected 心肌梗塞 / Myocardial Infarction")

    # 高脂血症（Hyperlipidemia）
    if lab_params.get("Total Cholesterol", 0) > 200 or lab_params.get("LDL-C", 0) > 130:
        diseases.append("高脂血症 / Hyperlipidemia")
        print("Debug: Detected 高脂血症 / Hyperlipidemia")

    # 心力衰竭（Heart Failure）
    if symptoms.get("Shortness of Breath", False) and lab_params.get("BNP", 0) > 100:
        diseases.append("心力衰竭 / Heart Failure")
        print("Debug: Detected 心力衰竭 / Heart Failure")

    if not diseases:
        diseases.append("无明显心血管疾病风险 / No significant cardiovascular disease risk detected")
        print("Debug: No diseases detected")

    print(f"Debug: Final Detected Diseases = {diseases}")
    return diseases

# 综合评估

def assess_with_huggingface(lang, *inputs):
    if not any(inputs):
        return "⚠️ 输入数据不足，无法完成评估 / Insufficient input data to complete the assessment."
    
    structured_inputs = inputs[:-1]
    free_text_input = inputs[-1]

    # Debug: Print structured inputs and free text input
    print(f"Debug: Structured Inputs = {structured_inputs}")
    print(f"Debug: Free Text Input = {free_text_input}")

    # Process structured inputs
    structured_result = assess(lang, *structured_inputs)

    # Debug: Print structured result
    print(f"Debug: Structured Result = {structured_result}")

    # Analyze free text
    huggingface_analysis = analyze_free_text(free_text_input)

    # Debug: Print Hugging Face analysis result
    print(f"Debug: Hugging Face Analysis = {huggingface_analysis}")

    # Extract symptoms, history, and lab parameters
    symptoms = {
        "Chest Pain": "是" in (structured_inputs[0] or "") if lang == "中文" else "Yes" in (structured_inputs[0] or ""),
        "Shortness of Breath": "是" in (structured_inputs[6] or "") if lang == "中文" else "Yes" in (structured_inputs[6] or ""),
    }
    history = {
        "Family History of Heart Disease": "是" in (structured_inputs[10] or "") if lang == "中文" else "Yes" in (structured_inputs[10] or ""),
    }
    lab_params = {
        "Systolic BP": structured_inputs[-6],
        "Diastolic BP": structured_inputs[-5],
        "LDL-C": structured_inputs[-4],
        "HDL-C": structured_inputs[-3],
        "Total Cholesterol": structured_inputs[-2],
        "Troponin I/T": structured_inputs[-1],
    }

    # Debug: Print extracted symptoms, history, and lab parameters
    print(f"Debug: Symptoms = {symptoms}")
    print(f"Debug: History = {history}")
    print(f"Debug: Lab Parameters = {lab_params}")

    # Evaluate diseases
    diseases = evaluate_cardiovascular_disease(symptoms, history, lab_params)

    # Debug: Print detected diseases
    print(f"Debug: Detected Diseases = {diseases}")

    # Detect conflicts
    conflict_detected = detect_conflicts(structured_result, huggingface_analysis)

    # Combine results
    combined_result = (
        f"### 来自问题判断 / Based on Structured Questions:\n{structured_result}\n\n"
        f"### 来自自由文字判断 / Based on Free Text Input:\n{huggingface_analysis}\n\n"
    )

    if conflict_detected:
        combined_result += (
            "⚠️ 检测到冲突 / Conflict Detected:\n"
            "结构化问题的答案与自由输入文字的分析结果存在冲突，请核实信息。\n\n"
        )

    combined_result += "### 疾病评估 / Disease Assessment:\n"
    combined_result += "\n".join(diseases)

    combined_result += "\n\n### 综合评估 / Combined Assessment:\n"
    combined_result += "综合考虑结构化问题和自由输入的结果，建议用户根据以上信息采取适当的行动。"

    # Debug: Print combined result
    print(f"Debug: Combined Result = {combined_result}")

    return combined_result

def assess_with_huggingface_1(lang, *inputs):
    if not any(inputs):
        return "⚠️ 输入数据不足，无法完成评估 / Insufficient input data to complete the assessment."
    structured_inputs = inputs[:-1]
    free_text_input = inputs[-1]

    # Debug: Print structured inputs and free text input
    print(f"Debug: Structured Inputs = {structured_inputs}")
    print(f"Debug: Free Text Input = {free_text_input}")


    # Process structured inputs
    structured_result = assess(lang, *structured_inputs)

    # Debug: Print structured result
    print(f"Debug: Structured Result = {structured_result}")

    # Analyze free text
    huggingface_analysis = analyze_free_text(free_text_input)

    # Debug: Print Hugging Face analysis result
    print(f"Debug: Hugging Face Analysis = {huggingface_analysis}")

    # Detect conflicts
    conflict_detected = detect_conflicts(structured_result, huggingface_analysis)

    # Combine results
    combined_result = (
        f"### 来自问题判断 / Based on Structured Questions:\n{structured_result}\n\n"
        f"### 来自自由文字判断 / Based on Free Text Input:\n{huggingface_analysis}\n\n"
    )

    if conflict_detected:
        combined_result += (
            "⚠️ 检测到冲突 / Conflict Detected:\n"
            "结构化问题的答案与自由输入文字的分析结果存在冲突，请核实信息。\n\n"
        )

    combined_result += "### 综合评估 / Combined Assessment:\n"
    combined_result += "综合考虑结构化问题和自由输入的结果，建议用户根据以上信息采取适当的行动。"

    print(f"Debug: Combined Result = {combined_result}")  # 调试输出
    return combined_result

# 评估结构化问题
def assess(lang, *inputs):
    risk_score = sum(1 for i in inputs if i == ("是" if lang == "中文" else "Yes"))
    print(f"Debug: Risk Score = {risk_score}")  # 调试输出
    if risk_score >= 5:
        return "🔴 高风险 / High Risk"
    elif risk_score >= 3:
        return "🟠 中风险 / Moderate Risk"
    else:
        return "🟢 低风险 / Low Risk"

# 创建语言标签页
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
    with gr.TabItem(lang):
        gr.Markdown(
            f"### 智能心血管评估系统 | Cardiovascular Assessment ({lang})"
        )

        # Symptom group with default values
        gr.Markdown("### 症状 / Symptoms" if lang == "中文" else "### Symptoms")
        symptom_fields = [
            gr.Radio(choices=yesno, value=L["no"], label=q)
            for q in symptom_questions
        ]

        # Medical history group with default values
        gr.Markdown("### 病史 / Medical History" if lang == "中文" else "### Medical History")
        history_fields = [
            gr.Radio(choices=yesno, value=L["no"], label=q)
            for q in history_questions
        ]

        # Lab parameters group with default values
        gr.Markdown("### 实验室参数 / Lab Parameters" if lang == "中文" else "### Lab Parameters")
        lab_fields = [
            gr.Number(
                label=f"{q} ({minv}-{maxv})",
                minimum=minv,
                maximum=maxv,
                value=val
            )
            for q, minv, maxv, val in L["nums"]
        ]

        # Free text input
        gr.Markdown("### 其他信息 / Additional Information" if lang == "中文" else "### Additional Information")
        free_text = gr.Textbox(
            label="📝 请提供其他相关信息 / Provide any additional relevant information" if lang == "中文" else "📝 Provide any additional relevant information",
            placeholder="请输入任何你想补充的健康信息……" if lang == "中文" else "Type here...",
            lines=3,
            max_lines=5,
            interactive=True
        )

        # Combine all fields
        fields = symptom_fields + history_fields + lab_fields + [free_text]

        # Output and submit button
        output = gr.Textbox(label="🩺 综合评估结果 / Combined Assessment Result" if lang == "中文" else "🩺 Combined Assessment Result")
        submit_button = gr.Button("提交评估 / Submit" if lang == "中文" else "Submit")
        reset_button = gr.Button("重置 / Reset" if lang == "中文" else "Reset")

        # Submit button functionality
        submit_button.click(
            fn=assess_with_huggingface,
            inputs=[gr.State(lang)] + fields,
            outputs=output
        )

        # Reset button functionality
        reset_button.click(
            fn=lambda lang: (
                [L["no"]] * len(symptom_fields) +
                [L["no"]] * len(history_fields) +
                [val for _, _, _, val in L["nums"]] +
                [""] +
                [""]
            ),
            inputs=[gr.State(lang)],
            outputs=symptom_fields + history_fields + lab_fields + [free_text, output]
        )

        # Debugging: Print the reset values
        print(f"Reset Values for Symptom Fields: {[L['no']] * len(symptom_fields)}")
        print(f"Reset Values for History Fields: {[L['no']] * len(history_fields)}")
        print(f"Reset Values for Number Fields: {[val for _, _, _, val in L['nums']]}")
        print(f"Reset Value for Free Text: {''}")
        print(f"Reset Value for Output: {''}")

def make_tab_1(lang):
    L = {
        "yes": "是", 
        "no": "否", 
        "nums": [
            ("收缩压 (mmHg)", 60, 220, 120),
            ("舒张压 (mmHg)", 40, 120, 80),
            ("低密度脂蛋白 (LDL-C, mg/dL)", 50, 200, 100),
            ("高密度脂蛋白 (HDL-C, mg/dL)", 20, 100, 50),
            ("总胆固醇 (Total Cholesterol, mg/dL)", 100, 300, 200),
            ("肌钙蛋白 (Troponin I/T, ng/mL)", 0, 50, 0.01)
        ]
    }
    yesno = [L["yes"], L["no"]]
    with gr.TabItem(lang):
        gr.Markdown(f"### 智能心血管评估系统 | Cardiovascular Assessment ({lang})")

        # 症状
        gr.Markdown("### 症状 / Symptoms")
        symptom_fields = [gr.Radio(choices=yesno, label=q) for q in [
            "胸痛是否在劳累时加重？", "是否为压迫感或紧缩感？", "是否持续超过5分钟？",
            "是否放射至肩/背/下巴？", "是否在休息后缓解？", "是否伴冷汗？",
            "是否呼吸困难？", "是否恶心或呕吐？", "是否头晕或晕厥？", "是否心悸？"
        ]]

        # 病史
        gr.Markdown("### 病史 / Medical History")
        history_fields = [gr.Radio(choices=yesno, label=q) for q in [
            "是否患有高血压？", "是否患糖尿病？", "是否有高血脂？", "是否吸烟？",
            "是否有心脏病家族史？", "近期是否有情绪压力？"
        ]]

        # 实验室参数
        gr.Markdown("### 实验室参数 / Lab Parameters")
        lab_fields = [
            gr.Number(label=q, minimum=minv, maximum=maxv, value=val)
            for q, minv, maxv, val in L["nums"]
        ]

        # 自由文本输入
        gr.Markdown("### 其他信息 / Additional Information")
        free_text = gr.Textbox(label="📝 请提供其他相关信息 / Provide any additional relevant information")

        # 组合所有字段
        fields = symptom_fields + history_fields + lab_fields + [free_text]

        # 输出和按钮
        output = gr.Textbox(label="🩺 综合评估结果 / Combined Assessment Result")
        submit_button = gr.Button("提交评估 / Submit")
        reset_button = gr.Button("重置 / Reset")

        # 提交按钮功能
        submit_button.click(
            fn=assess_with_huggingface,
            inputs=[gr.State(lang)] + fields,
            outputs=output
        )

        # 重置按钮功能
        reset_button.click(
            fn=lambda: (
                [None] * len(symptom_fields) +
                [None] * len(history_fields) +
                [None] * len(lab_fields) +
                [""],  # Reset free text
                ""     # Reset output
            ),
            inputs=None,
            outputs=symptom_fields + history_fields + lab_fields + [free_text, output]
        )

# 启动 Gradio 应用
if __name__ == "__main__":
    with gr.Blocks() as app:
        gr.Markdown("## 🌐 智能心血管评估系统 | Bilingual Cardiovascular Assistant")
        with gr.Tabs():
            make_tab("中文")
            make_tab("English")
        app.launch(share=True)