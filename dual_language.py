import gradio as gr
from transformers import pipeline

# Load the Hugging Face pipeline for text classification
text_analysis_pipeline = pipeline("text-classification", model="dmis-lab/biobert-base-cased-v1.1")

# Define label mapping
LABEL_MAPPING = {
    "LABEL_0": "低风险 / Low Risk",
    "LABEL_1": "中风险 / Moderate Risk",
    "LABEL_2": "高风险 / High Risk"
}

# Analyze free text using Hugging Face model
def analyze_free_text(free_text):
    if not free_text.strip():
        return "无额外信息 / No additional information provided."

    try:
        results = text_analysis_pipeline(free_text)
        analysis = "\n".join([
            f"{LABEL_MAPPING.get(label['label'], label['label'])}: {label['score']:.2f}"
            for label in results
        ])
        return f"分析结果 / Analysis Results:\n{analysis}"
    except Exception as e:
        return f"无法分析自由文本信息 / Unable to analyze free text information: {e}"

# Detect conflicts between structured questions and free text analysis
def detect_conflicts(structured_result, huggingface_analysis):
    if "低风险" in structured_result and "高风险" in huggingface_analysis:
        return True
    if "高风险" in structured_result and "低风险" in huggingface_analysis:
        return True
    return False

# Assess structured questions and combine with free text analysis
def assess_with_huggingface(lang, *inputs):
    structured_inputs = inputs[:-1]
    free_text_input = inputs[-1]

    structured_result = assess(lang, *structured_inputs)
    huggingface_analysis = analyze_free_text(free_text_input)
    conflict_detected = detect_conflicts(structured_result, huggingface_analysis)

    combined_result = (
        f"### 来自问题判断 / Based on Structured Questions:\n{structured_result}\n\n"
        f"### 来自自由文字判断 / Based on Free Text Input:\n{huggingface_analysis}\n\n"
    )

    if conflict_detected:
        combined_result += (
            "\u26a0\ufe0f 检测到冲突 / Conflict Detected:\n"
            "结构化问题的答案与自由输入文字的分析结果存在冲突，请核实信息\n\n"
        )

    combined_result += "### 综合评估 / Combined Assessment:\n"
    combined_result += "综合考虑结构化问题和自由输入的结果，建议用户根据以上信息采取适当的行动。"

    return combined_result

# Example structured question assessment function
def assess(lang, *inputs):
    risk_score = sum(1 for i in inputs if i == ("是" if lang == "中文" else "Yes"))
    if risk_score >= 5:
        return "🔴 高风险 / High Risk"
    elif risk_score >= 3:
        return "�\dfe0 中风险 / Moderate Risk"
    else:
        return "🟢 低风险 / Low Risk"

# Create a tab for each language
def make_tab(lang):
    if lang == "中文":
        L = {"yes": "是", "no": "否", "nums": [("收缩压 (mmHg)", 60, 220, 120)]}
    else:
        L = {"yes": "Yes", "no": "No", "nums": [("Systolic BP (mmHg)", 60, 220, 120)]}

    yesno = [L["yes"], L["no"]]

    with gr.TabItem(lang):
        gr.Markdown(f"### Cardiovascular Assessment ({lang})")

        gr.Markdown("### Symptoms")
        symptom_fields = [gr.Radio(choices=yesno, label=q) for q in [
            "Does chest pain worsen with exertion?" if lang != "中文" else "胸痛是否在劳类时加重？",
            "Is it a pressing or squeezing sensation?" if lang != "中文" else "是否为压挤感或紧缩感？",
            "Does it last longer than 5 minutes?" if lang != "中文" else "是否持续超过5分钟？",
            "Does it radiate to the shoulder/back/jaw?" if lang != "中文" else "是否放射至肩/背/下巴？",
            "Does it improve with rest?" if lang != "中文" else "是否在休息后缓解？",
            "Is it accompanied by cold sweats?" if lang != "中文" else "是否伴冷汗？",
            "Is there shortness of breath?" if lang != "中文" else "是否呼吸困难？",
            "Is there nausea or vomiting?" if lang != "中文" else "是否恼恼或呕吐？",
            "Is there dizziness or fainting?" if lang != "中文" else "是否头晕或晕剧？",
            "Is there heart palpitations?" if lang != "中文" else "是否心惊？"
        ]]

        gr.Markdown("### Medical History")
        history_fields = [gr.Radio(choices=yesno, label=q) for q in [
            "Do you have high blood pressure?" if lang != "中文" else "是否患有高血压？",
            "Do you have diabetes?" if lang != "中文" else "是否患糖尿病？",
            "Do you have high cholesterol?" if lang != "中文" else "是否有高血脂？",
            "Do you smoke?" if lang != "中文" else "是否吸烟？",
            "Is there a family history of heart disease?" if lang != "中文" else "是否有心脏病家族史？",
            "Have you experienced recent emotional stress?" if lang != "中文" else "迟期是否有情绪压力？"
        ]]

        gr.Markdown("### Lab Parameters")
        lab_fields = [
            gr.Number(label=q, minimum=minv, maximum=maxv, value=val)
            for q, minv, maxv, val in L["nums"]
        ]

        gr.Markdown("### Additional Information")
        free_text = gr.Textbox(label="📝 Provide any additional relevant information" if lang != "中文" else "📝 请提供其他相关信息")

        fields = symptom_fields + history_fields + lab_fields + [free_text]

        with gr.Group():
            output = gr.Textbox(label="🩺 Combined Assessment Result", key=f"output_{lang}")
            submit_button = gr.Button("Submit", key=f"submit_{lang}")
            reset_button = gr.Button("Reset", key=f"reset_{lang}")

        submit_button.click(
            fn=assess_with_huggingface,
            inputs=[gr.State(lang)] + fields,
            outputs=output
        )

        reset_button.click(
            fn=lambda: (
                [None] * len(symptom_fields) +
                [None] * len(history_fields) +
                [None] * len(lab_fields) +
                [""],
                ""
            ),
            inputs=None,
            outputs=symptom_fields + history_fields + lab_fields + [free_text, output]
        )

if __name__ == "__main__":
    with gr.Blocks() as app:
        gr.Markdown("## 🌐 智能心血管评估系统 | Bilingual Cardiovascular Assistant")
        with gr.Tabs():
            make_tab("\u4e2d\u6587")
            make_tab("English")
        app.launch(share=True)
