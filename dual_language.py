import openai  # 确保已安装 openai 库
import gradio as gr
from transformers import pipeline

# 加载 Hugging Face 模型管道
# 使用 BioBERT 模型进行文本分类
text_analysis_pipeline = pipeline("text-classification", model="dmis-lab/biobert-base-cased-v1.1")

# 调用 Hugging Face 模型分析自由文本
def analyze_free_text(free_text):
    if not free_text.strip():
        return "无额外信息 / No additional information provided."
    
    try:
        # 使用 Hugging Face 模型分析自由文本
        results = text_analysis_pipeline(free_text)
        # 格式化分析结果
        analysis = "\n".join([f"{label['label']}: {label['score']:.2f}" for label in results])
        return f"分析结果 / Analysis Results:\n{analysis}"
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