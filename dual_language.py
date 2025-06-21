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
            ("LDL胆固醇 (1.0–7.0 mmol/L，可选)", 1.0, 7.0, 2.8)

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
            ("LDL Cholesterol (mmol/L, optional)", 1.0, 7.0, 2.8)
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


def enforce_word_limit(text):
    words = text.strip().split()
    if len(words) > 500:
        return " ".join(words[:500])
    return text

# 推理函数
def assess(lang, *answers):
    print(f"[LOG] answers: {answers}")
    L = content[lang]
    yn = L["yes"]
    p = [a == yn for a in answers[:16]]
    bp, glu, ldl = answers[16], answers[17], answers[18]
    reasons = []
    score = sum(p)
    if bp and bp >= 140:
        score += 1; reasons.append("血压偏高" if lang == "中文" else "High BP")
    if glu and glu >= 7.0:
        score += 1; reasons.append("血糖偏高" if lang == "中文" else "High Glucose")
    if ldl and ldl >= 3.4:
        score += 1; reasons.append("LDL偏高" if lang == "中文" else "High LDL")
    print(f"Score: {score}, Reasons: {reasons}")
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
        reasons.append("Non-specific symptoms" if lang ==
                       "English" else "症状不典型")

    freetext = answers[-1]
    print(f"[LOG] Input Free Text ({lang}): {freetext}")
    print(f"[LOG] Score before free text: {score}")
    if freetext:
        keywords_en = ["pain", "sweat", "dizzy", "palpitation", "vomit"]
        keywords_zh = ["疼痛", "出汗", "头晕", "心悸", "呕吐"]
        found = []
        if lang == "English":
            for kw in keywords_en:
                if kw in freetext.lower():
                    found.append(kw)
            if found:
                score += 1
                reasons.append(f"Free text mentions: {', '.join(found)}")
        else:
            for kw in keywords_zh:
                if kw in freetext:
                    found.append(kw)
                    score += 1
    print(f"[LOG] Score after free text: {score}")
    print(f"[LOG] Reasoning ({lang}): {reasons}")
    result = L["report"].format(
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        level, advice, d,
        ", ".join(reasons) if lang == "English" else "，".join(reasons),
        L["diseases"][d]
    )

    # Log the output result
    print(f"[LOG] Output Result ({lang}): {result}")

    return result

# 构建标签页
def make_tab(lang):
    L = content[lang]
    yesno = [L["yes"], L["no"]]
    with gr.TabItem(lang):
        gr.Markdown(f"### {L['title']}")
        gr.Markdown(L["desc"])

        fields = [gr.Radio(choices=yesno, label=q) for q in L["inputs"]]
        print(f'{lang} inputs: {len(fields)} questions')
        for q, minv, maxv, val in L["nums"]:
            fields.append(gr.Number(label=q, minimum=minv, maximum=maxv, value=val))

        freetext = gr.Textbox(
            label="Free Text" if lang == "English" else "自由文本",
            lines=10,
            placeholder="Enter up to 500 words..." if lang == "English" else "最多输入500个单词...",
            elem_id=f"freetext_{lang}"
        )

        output = gr.Textbox(label="🩺 结果 / Result")
        gr.Button("提交评估 / Submit").click(
            fn=assess,
            inputs=[gr.State(lang)] + fields +
            [freetext],  # Add freetext to inputs
            outputs=output
        )

# 启动 Gradio 应用
with gr.Blocks() as app:
    gr.Markdown("## 🌐 智能心血管评估系统 | Bilingual Cardiovascular Assistant")
    with gr.Tabs():
        make_tab("中文")
        make_tab("English")

if __name__ == "__main__":
    app.launch(share=True)

