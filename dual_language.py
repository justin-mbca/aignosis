import gradio as gr
from datetime import datetime

# 双语模板
content = {
    "中文": {
        "title": "🫀 智能心血管风险评估助手",
        "desc": "请填写以下信息，系统将智能预测风险等级与可能疾病类型。",
        "yes": "是", "no": "否",
        "inputs": [
            "胸痛是否在劳累时加重？", "是否为压迫感或紧缩感？", "是否持续超过5分钟？",
            "是否放射至肩/背/下巴？", "是否在休息后缓解？", "是否伴冷汗？",
            "是否呼吸困难？", "是否恶心或呕吐？", "是否头晕或晕厥？", "是否心悸？",
            "是否患有高血压？", "是否患糖尿病？", "是否有高血脂？", "是否吸烟？",
            "是否有心脏病家族史？", "近期是否有情绪压力？"
        ],
        "nums": [
            ("收缩压 (60–220 mmHg，可选)", 60, 220, 120),
            ("空腹血糖 (2.0–20.0 mmol/L)", 2.0, 20.0, 5.5),
            ("LDL胆固醇 (1.0–7.0 mmol/L，可选)", 1.0, 7.0, 2.8),
            ("肌钙蛋白 (Troponin, ng/mL)", 0.0, 50.0, 0.01),
            ("肌酸激酶同工酶 (CK-MB, ng/mL)", 0.0, 50.0, 1.0)
        ],
        "results": {
            "high": "🔴 高风险", "mid": "🟠 中风险", "low": "🟢 低风险",
            "advice_h": "请立即就医，排除急性心血管事件。",
            "advice_m": "建议尽快进行心电图和医学评估。",
            "advice_l": "风险较低，请保持健康生活方式。"
        },
        "diseases": {
            "稳定型心绞痛": "活动诱发胸痛、休息缓解，提示冠状动脉供血不足。",
            "急性心肌梗死": "持续胸痛 + 放射痛 + 冷汗，提示心肌梗死风险。",
            "心力衰竭": "呼吸困难 + 头晕/水肿，提示泵血功能下降。",
            "心律失常": "心悸 + 晕厥，提示节律异常。",
            "非典型胸痛": "症状不典型，建议排除非心源性疾病。"
        },
        "report": "📝 时间：{}\n\n🧮 风险等级：{}\n📌 建议：{}\n\n🫀 疾病推测：{}\n🧠 推理依据：{}\n📘 疾病说明：\n{}"
    },
    "English": {
        "title": "🫀 AI Cardiovascular Risk Estimator",
        "desc": "Please complete the questions below. The AI will estimate your risk level and possible disease type.",
        "yes": "Yes", "no": "No",
        "inputs": [
            "Chest pain triggered by exertion?", "Is it pressure or tightness?", "Lasts more than 5 minutes?",
            "Radiates to shoulder/back/jaw?", "Relieved by rest?", "With cold sweat?",
            "Shortness of breath?", "Nausea or vomiting?", "Dizziness or fainting?", "Palpitations?",
            "History of hypertension?", "History of diabetes?", "High cholesterol?", "Smoking?",
            "Family history of heart disease?", "Recent emotional stress?"
        ],
        "nums": [
            ("Systolic BP (mmHg, optional)", 60, 220, 120),
            ("Fasting Glucose (mmol/L)", 2.0, 20.0, 5.5),
            ("LDL Cholesterol (mmol/L, optional)", 1.0, 7.0, 2.8),
            ("Troponin (ng/mL)", 0.0, 50.0, 0.01),
            ("CK-MB (ng/mL)", 0.0, 50.0, 1.0)
        ],
        "results": {
            "high": "🔴 High Risk", "mid": "🟠 Moderate Risk", "low": "🟢 Low Risk",
            "advice_h": "Seek emergency medical care immediately.",
            "advice_m": "Recommend ECG and medical evaluation.",
            "advice_l": "Low risk. Maintain healthy lifestyle and monitor."
        },
        "diseases": {
            "Stable Angina": "Exertional chest pain relieved by rest. Suggests coronary narrowing.",
            "Acute Myocardial Infarction": "Persistent pain with radiation and sweat suggests heart attack.",
            "Heart Failure": "Shortness of breath with dizziness or edema. Indicates cardiac dysfunction.",
            "Arrhythmia": "Palpitations with fainting suggests rhythm abnormality.",
            "Atypical Chest Pain": "Non-specific symptoms. Consider other possible causes."
        },
        "report": "📝 Time: {}\n\n🧮 Risk Level: {}\n📌 Recommendation: {}\n\n🫀 Possible Disease: {}\n🧠 Reasoning: {}\n📘 Explanation:\n{}"
    }
}

# 推理函数
def assess(lang, *answers):
    L = content[lang]
    yn = L["yes"]
    p = [a == yn for a in answers[:16]]
    bp, glu, ldl, troponin, ck_mb = answers[16], answers[17], answers[18], answers[19], answers[20]
    reasons = []
    score = sum(p)
    if bp and bp >= 140:
        score += 1; reasons.append("血压偏高" if lang == "中文" else "High BP")
    if glu and glu >= 7.0:
        score += 1; reasons.append("血糖偏高" if lang == "中文" else "High Glucose")
    if ldl and ldl >= 3.4:
        score += 1; reasons.append("LDL偏高" if lang == "中文" else "High LDL")
    if troponin and troponin > 0.04:
        score += 2; reasons.append("肌钙蛋白升高" if lang == "中文" else "Elevated Troponin")
    if ck_mb and ck_mb > 5.0:
        score += 2; reasons.append("肌酸激酶同工酶升高" if lang == "中文" else "Elevated CK-MB")

    r = L["results"]
    if score >= 9: level, advice = r["high"], r["advice_h"]
    elif score >= 5: level, advice = r["mid"], r["advice_m"]
    else: level, advice = r["low"], r["advice_l"]

    # 疾病判断
    if p[0] and p[1] and p[4] and not p[3] and not p[5]:
        d = "Stable Angina" if lang == "English" else "稳定型心绞痛"
        reasons.append("Effort + rest relief" if lang == "English" else "劳力性胸痛，休息缓解")
    elif p[2] and p[3] and p[5]:
        d = "Acute Myocardial Infarction" if lang == "English" else "急性心肌梗死"
        reasons.append("Persistent + radiation + sweat" if lang == "English" else "持续胸痛 + 放射痛 + 冷汗")
    elif p[6] and (p[7] or p[8]):
        d = "Heart Failure" if lang == "English" else "心力衰竭"
        reasons.append("Dyspnea + dizziness" if lang == "English" else "呼吸困难 + 头晕")
    elif p[9] and p[8]:
        d = "Arrhythmia" if lang == "English" else "心律失常"
        reasons.append("Palpitations + fainting" if lang == "English" else "心悸 + 晕厥")
    else:
        d = "Atypical Chest Pain" if lang == "English" else "非典型胸痛"
        reasons.append("Non-specific symptoms" if lang == "English" else "症状不典型")

    return L["report"].format(
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        level, advice, d,
        ", ".join(reasons) if lang == "English" else "，".join(reasons),
        L["diseases"][d]
    )

# 构建标签页
def make_tab(lang):
    L = content[lang]
    yesno = [L["yes"], L["no"]]
    with gr.TabItem(lang):
        gr.Markdown(f"### {L['title']}")
        gr.Markdown(L["desc"])

        # Add custom CSS for responsive grid layout
        gr.HTML("""
        <style>
            .grid-container {
                display: grid;
                gap: 16px;
            }
            /* Default: 2 columns */
            .grid-container {
                grid-template-columns: repeat(2, 1fr);
            }
            /* Medium screens: 3 columns */
            @media (min-width: 768px) {
                .grid-container {
                    grid-template-columns: repeat(3, 1fr);
                }
            }
            /* Large screens: 4 columns */
            @media (min-width: 1200px) {
                .grid-container {
                    grid-template-columns: repeat(4, 1fr);
                }
            }
        </style>
        """)

        # 症状分组
        gr.Markdown("### 症状 / Symptoms")
        gr.HTML('<div class="grid-container">')
        symptom_fields = [gr.Radio(choices=yesno, label=q) for q in [
            "胸痛是否在劳累时加重？", "是否为压迫感或紧缩感？", "是否持续超过5分钟？",
            "是否放射至肩/背/下巴？", "是否在休息后缓解？", "是否伴冷汗？",
            "是否呼吸困难？", "是否恶心或呕吐？", "是否头晕或晕厥？", "是否心悸？"
        ]]
        gr.HTML('</div>')

        # 病史分组
        gr.Markdown("### 病史 / Medical History")
        gr.HTML('<div class="grid-container">')
        history_fields = [gr.Radio(choices=yesno, label=q) for q in [
            "是否患有高血压？", "是否患糖尿病？", "是否有高血脂？", "是否吸烟？",
            "是否有心脏病家族史？", "近期是否有情绪压力？"
        ]]
        gr.HTML('</div>')

      # 实验室参数分组
        gr.Markdown("### 实验室参数 / Lab Parameters")
        lab_fields = [
            gr.Number(label=q, minimum=minv, maximum=maxv, value=val)
            for q, minv, maxv, val in L["nums"]
        ]

        # 合并所有字段
        fields = symptom_fields + history_fields + lab_fields

        # Add custom CSS for fixed 4-column layout
        gr.HTML("""
        <style>
            .grid-container {
                display: grid;
                grid-template-columns: repeat(4, 1fr); /* Fixed 4 columns */
                gap: 16px;
            }
        </style>
        """)
        # Add custom CSS for fixed 4-column layout
        gr.HTML("""
        <style>
            .grid-container {
                display: grid;
                grid-template-columns: repeat(4, 1fr); /* Fixed 4 columns */
                gap: 16px;
            }
        </style>
        """)

        # Wrap all fields in a grid container
        with gr.Row():
            with gr.HTML('<div class="grid-container">'):
                for field in fields:
                    field  # Add each field to the grid container
            gr.HTML('</div>')

        # 输出和提交按钮
        output = gr.Textbox(label="🩺 结果 / Result")
        gr.Button("提交评估 / Submit").click(
            fn=assess,
            inputs=[gr.State(lang)] + fields,
            outputs=output
        )

 # 启动 Gradio 应用
if __name__ == "__main__":
    with gr.Blocks() as app:
        gr.Markdown("## 🌐 智能心血管评估系统 | Bilingual Cardiovascular Assistant")
        with gr.Tabs():
            make_tab("中文")
            make_tab("English")
        app.launch(share=True)  # Move the launch call inside the context