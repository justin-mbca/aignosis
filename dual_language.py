import openai  # 确保已安装 openai 库
import gradio as gr

# 使用 LLM 分析自由文本并结合结构化问题
def assess_with_llm(lang, *inputs):
    # 分离结构化输入和自由文本
    structured_inputs = inputs[:-1]
    free_text_input = inputs[-1]

    # 调用 assess 函数处理结构化输入
    structured_result = assess(lang, *structured_inputs)

    # 调用 LLM 分析自由文本
    llm_analysis = analyze_free_text(free_text_input)

    # 综合结果
    combined_result = (
        f"### 来自问题判断 / Based on Structured Questions:\n{structured_result}\n\n"
        f"### 来自自由文字判断 / Based on Free Text Input:\n{llm_analysis}\n\n"
        f"### 综合评估 / Combined Assessment:\n"
        f"综合考虑结构化问题和自由输入的结果，建议用户根据以上信息采取适当的行动。"
    )
    return combined_result

# 调用 LLM 分析自由文本
def analyze_free_text(free_text):
    if not free_text.strip():
        return "无额外信息 / No additional information provided."
    
    # 调用 OpenAI GPT API 分析自由文本
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 使用最新的 ChatGPT 模型
            messages=[
                {"role": "system", "content": "你是一个心血管健康助手，请分析用户提供的信息并提取与心血管健康相关的内容。"},
                {"role": "user", "content": free_text}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"无法分析自由文本信息 / Unable to analyze free text information: {e}"
# 示例结构化问题评估函数
def assess(lang, *inputs):
    # 示例逻辑：根据结构化问题计算风险等级
    risk_score = sum(1 for i in inputs if i == "是")  # 假设“是”表示风险
    if risk_score >= 5:
        return "🔴 高风险 / High Risk"
    elif risk_score >= 3:
        return "🟠 中风险 / Moderate Risk"
    else:
        return "🟢 低风险 / Low Risk"

# 构建 Gradio 界面
def make_tab(lang):
    L = {"yes": "是", "no": "否", "nums": [("收缩压 (mmHg)", 60, 220, 120)]}
    yesno = [L["yes"], L["no"]]
    with gr.TabItem(lang):
        gr.Markdown(f"### 智能心血管评估系统 | Cardiovascular Assessment ({lang})")

        # 症状分组
        gr.Markdown("### 症状 / Symptoms")
        symptom_fields = [gr.Radio(choices=yesno, label=q) for q in [
            "胸痛是否在劳累时加重？", "是否为压迫感或紧缩感？", "是否持续超过5分钟？",
            "是否放射至肩/背/下巴？", "是否在休息后缓解？", "是否伴冷汗？",
            "是否呼吸困难？", "是否恶心或呕吐？", "是否头晕或晕厥？", "是否心悸？"
        ]]

        # 病史分组
        gr.Markdown("### 病史 / Medical History")
        history_fields = [gr.Radio(choices=yesno, label=q) for q in [
            "是否患有高血压？", "是否患糖尿病？", "是否有高血脂？", "是否吸烟？",
            "是否有心脏病家族史？", "近期是否有情绪压力？"
        ]]

        # 实验室参数分组
        gr.Markdown("### 实验室参数 / Lab Parameters")
        lab_fields = [
            gr.Number(label=q, minimum=minv, maximum=maxv, value=val)
            for q, minv, maxv, val in L["nums"]
        ]

        # 自由文本输入
        gr.Markdown("### 其他信息 / Additional Information")
        free_text = gr.Textbox(label="📝 请提供其他相关信息 / Provide any additional relevant information")

        # 合并所有字段
        fields = symptom_fields + history_fields + lab_fields + [free_text]

        # 输出和提交按钮
        output = gr.Textbox(label="🩺 综合评估结果 / Combined Assessment Result")
        gr.Button("提交评估 / Submit").click(
            fn=assess_with_llm,
            inputs=[gr.State(lang)] + fields,
            outputs=output
        )

# 启动 Gradio 应用
if __name__ == "__main__":
    openai.api_key = "your_openai_api_key"  # 替换为您的 OpenAI API 密钥
    with gr.Blocks() as app:
        gr.Markdown("## 🌐 智能心血管评估系统 | Bilingual Cardiovascular Assistant")
        with gr.Tabs():
            make_tab("中文")
            make_tab("English")
        app.launch(share=True)