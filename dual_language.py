import gradio as gr
from transformers import pipeline

# Load the Hugging Face pipeline for text classification
# Replace "dmis-lab/biobert-base-cased-v1.1" with a fine-tuned model if needed
text_analysis_pipeline = pipeline("text-classification", model="dmis-lab/biobert-base-cased-v1.1")
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
        # 使用 Hugging Face 模型分析自由文本
        results = text_analysis_pipeline(free_text)
        
        # 转换标签为文字描述
        analysis = "\n".join([
            f"{LABEL_MAPPING.get(label['label'], label['label'])}: {label['score']:.2f}"
            for label in results
        ])
        return f"分析结果 / Analysis Results:\n{analysis}"
    except Exception as e:
        return f"无法分析自由文本信息 / Unable to analyze free text information: {e}"
    
def detect_conflicts(structured_result, huggingface_analysis):
    """
    检测结构化问题的结果和自由输入文字的分析结果是否存在冲突。
    """
    # 示例逻辑：如果结构化问题的结果是低风险，但自由文本分析显示高风险，则认为存在冲突
    if "低风险" in structured_result and "高风险" in huggingface_analysis:
        return True
    if "高风险" in structured_result and "低风险" in huggingface_analysis:
        return True

    # 如果没有检测到冲突
    return False    #最近一周经常感到胸闷，尤其是在爬楼梯时。持续时间大约5分钟，休息后会缓解。家族中父亲有冠心病史。

# Assess structured questions and combine with free text analysis
def assess_with_huggingface(lang, *inputs):
    # Separate structured inputs and free text
    structured_inputs = inputs[:-1]
    free_text_input = inputs[-1]

    # Process structured inputs
    structured_result = assess(lang, *structured_inputs)

    # Analyze free text
    huggingface_analysis = analyze_free_text(free_text_input)

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

    return combined_result

# Example structured question assessment function
def assess(lang, *inputs):
    # Example logic: Calculate risk level based on structured questions
    risk_score = sum(1 for i in inputs if i == "是")  # Assume "是" indicates risk
    if risk_score >= 5:
        return "🔴 高风险 / High Risk"
    elif risk_score >= 3:
        return "🟠 中风险 / Moderate Risk"
    else:
        return "🟢 低风险 / Low Risk"

def make_tab(lang):
    L = {"yes": "是", "no": "否", "nums": [("收缩压 (mmHg)", 60, 220, 120)]}
    yesno = [L["yes"], L["no"]]
    with gr.TabItem(lang):
        gr.Markdown(f"### 智能心血管评估系统 | Cardiovascular Assessment ({lang})")

        # Symptom group
        gr.Markdown("### 症状 / Symptoms")
        symptom_fields = [gr.Radio(choices=yesno, label=q) for q in [
            "胸痛是否在劳累时加重？", "是否为压迫感或紧缩感？", "是否持续超过5分钟？",
            "是否放射至肩/背/下巴？", "是否在休息后缓解？", "是否伴冷汗？",
            "是否呼吸困难？", "是否恶心或呕吐？", "是否头晕或晕厥？", "是否心悸？"
        ]]

        # Medical history group
        gr.Markdown("### 病史 / Medical History")
        history_fields = [gr.Radio(choices=yesno, label=q) for q in [
            "是否患有高血压？", "是否患糖尿病？", "是否有高血脂？", "是否吸烟？",
            "是否有心脏病家族史？", "近期是否有情绪压力？"
        ]]

        # Lab parameters group
        gr.Markdown("### 实验室参数 / Lab Parameters")
        lab_fields = [
            gr.Number(label=q, minimum=minv, maximum=maxv, value=val)
            for q, minv, maxv, val in L["nums"]
        ]

        # Free text input
        gr.Markdown("### 其他信息 / Additional Information")
        free_text = gr.Textbox(label="📝 请提供其他相关信息 / Provide any additional relevant information")

        # Combine all fields
        fields = symptom_fields + history_fields + lab_fields + [free_text]

        # Output and submit button
        output = gr.Textbox(label="🩺 综合评估结果 / Combined Assessment Result")
        submit_button = gr.Button("提交评估 / Submit")
        reset_button = gr.Button("重置 / Reset")  # Add reset button

        # Submit button functionality
        submit_button.click(
            fn=assess_with_huggingface,  # Function to process inputs
            inputs=[gr.State(lang)] + fields,
            outputs=output
        )

        # Reset button functionality
        reset_button.click(
            fn=lambda: (
                [None] * len(symptom_fields) +  # Reset all Radio fields
                [None] * len(history_fields) +  # Reset all Radio fields
                [None] * len(lab_fields) +      # Reset all Number fields
                [""],                          # Reset the free text field
                ""                             # Reset the output field
            ),
            inputs=None,
            outputs=fields + [output]  # Reset all inputs and the output
        )
# Launch Gradio app
if __name__ == "__main__":
    with gr.Blocks() as app:
        gr.Markdown("## 🌐 智能心血管评估系统 | Bilingual Cardiovascular Assistant")
        with gr.Tabs():
            make_tab("中文")
            make_tab("English")
        app.launch(share=True)