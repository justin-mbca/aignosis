import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def summarize_model_outputs_llm(model_outputs, language="中文"):
    """
    Summarize model outputs using OpenAI GPT, returning the JSON summary string.
    Args:
        model_outputs: list of dicts with model results
        language: "中文" or "English"
    Returns:
        str: JSON summary from LLM
    """
    # def format_model_outputs(outputs):
    #     formatted = ""
    #     for item in outputs:
    #         probs = "\n    ".join([f"🔹 {k}: {v:.2f}" for k, v in item["probabilities"].items()])
    #         formatted += (
    #             f"- **{item['model_name']}**:\n"
    #             f"  **风险等级:** {item['most_likely']}\n"
    #             f"  **概率分布:**\n"
    #             f"    {probs}\n"
    #             f"  **模型解释:** {item['explanation']}\n\n"
    #         )
    #     return formatted

    # formatted_text = format_model_outputs(model_outputs)
    # print(f"Model outputs: {model_outputs}")
    formatted_text = model_outputs

    if language == "中文":
        prompt = f"""
你是一个医学风险分析助手。

模型输出如下：
{formatted_text}

请根据上方的模型输出内容，生成一份详细的用户报告，但不要重复或引用模型输出的原文，仅根据其信息进行总结和建议，内容包括：
1. 总体风险等级判断
2. 各模型之间的一致性或差异分析
3. 风险等级建议
4. 针对用户的具体行动建议（如是否需要就医、紧急程度、可否观察等待、应准备哪些信息等）
请只用中文输出。
请确保内容结构、详细程度与英文报告保持一致。
"""
    else:
        prompt = f"""
You are a medical risk analysis assistant.

The model output is as follows:
{formatted_text}

Based on the above model output, generate a detailed user report, but do not repeat or quote the model output verbatim. Use its information to summarize and provide recommendations, including:
1. Overall risk level assessment
2. Consistency or differences among models
3. Risk level recommendation
4. Specific user action suggestions (e.g., whether to see a doctor, urgency, whether to wait and observe, what information to prepare, etc.)
Please output only in English.
Make sure the structure and level of detail match the Chinese report.
"""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Example usage for standalone test
    
    model_outputs_chinese = """
## 🩺 综合风险等级
🔹 **高风险**

## 📊 模型概率分布
### 🔸 BioBERT
- 高风险: 0.38
### 🔸 PubMedBERT
- 高风险: 0.51
### 🔸 ClinicalBERT
- 高风险: 0.67

## ❤️ HEART评分: 1分 (低风险)
## ⚖️ 加权风险分数
- 低风险: 0.000
- 中风险: 0.000
- 高风险: 0.521

## 🩺 临床建议
- 这是紧急情况，请立即就医。

## 💬 模型说明
### BioBERT
BioBERT 是一个专门针对生物医学文本训练的模型，适用于分析医学相关的文本。

### PubMedBERT
PubMedBERT 是基于 PubMed 数据训练的模型，专注于生物医学文献的理解。

### ClinicalBERT
ClinicalBERT 是针对临床文本（如电子病历）优化的模型，适合分析患者相关的临床数据。


## 📝 输入摘要
### 📝 用户输入:

#### 🩺 症状:
🔹 胸痛是否在劳累时加重？: 否
🔹 是否为压迫感或紧缩感？: 否
🔹 是否持续超过5分钟？: 是
🔹 是否放射至肩/背/下巴？: 否
🔹 是否在休息后缓解？: 否
🔹 是否伴冷汗？: 是
🔹 是否呼吸困难？: 否
🔹 是否恶心或呕吐？: 否
🔹 是否头晕或晕厥？: 否
🔹 是否心悸？: 否

#### 🏥 病史:
🔹 是否患有高血压？: 否
🔹 是否患糖尿病？: 是
🔹 是否有高血脂？: 否
🔹 是否吸烟？: 是
🔹 是否有心脏病家族史？: 否
🔹 近期是否有情绪压力？: 否

#### 🧪 实验室参数:
🔹 收缩压 (mmHg): 120
🔹 舒张压 (mmHg): 80
🔹 低密度脂蛋白胆固醇 (mg/dL): 171.4
🔹 高密度脂蛋白胆固醇 (mg/dL): 78.15
🔹 总胆固醇 (mg/dL): 266.5
🔹 肌钙蛋白 (Troponin I/T, ng/mL): 0.01
### 上传文件内容解析
{
  "癌胚抗原 (CEA)": "3.22 ng/ml (≤5)",
  "甲胎蛋白": "3.52 ng/ml (≤7)",
  "高密度脂蛋白胆固醇": "78.15 mg/dL (>40)",
  "低密度脂蛋白胆固醇": "171.4 mg/dL (<130) ↑",
  "甘油三酯": "110.7 mg/dL (<150)",
  "总胆固醇": "266.5 mg/dL (<200) ↑",
  "尿素": "37.64 mg/dL (18.63–52.85)",
  "总二氧化碳": "26.8 mEq/L (22.0–29.0)",
  "尿酸": "3.97 mg/dL (2.61–6.00)",
  "肌酐": "0.71 mg/dL (0.46–0.92)"
}

文件覆盖实验室参数： 高密度脂蛋白胆固醇 (mg/dL):78.15 替换  50
文件覆盖实验室参数： 低密度脂蛋白胆固醇 (mg/dL):171.4 替换  100
文件覆盖实验室参数： 总胆固醇 (mg/dL):266.5 替换  200
"""
    language = "English"  # or "中文"
    model_outputs_english = """
## 🩺 Overall risk
🔹 **High Risk**

## 📊 Model Probability Distribution
### 🔸 BioBERT
- Moderate Risk: 0.37
### 🔸 PubMedBERT
- High Risk: 0.52
### 🔸 ClinicalBERT
- High Risk: 0.64

## ❤️ HEART Score: 1 points (Low Risk)
## ⚖️ Weighted Risk Scores
- Low Risk: 0.000
- Moderate Risk: 0.110
- High Risk: 0.398

## 🩺 Clinical Recommendations
- This is an emergency. Please seek medical attention immediately.

## 💬 Model Explanation
### BioBERT
BioBERT is a model pre-trained on biomedical text, suitable for analyzing medical-related content.

### PubMedBERT
PubMedBERT is trained on PubMed data and focuses on understanding biomedical literature.

### ClinicalBERT
ClinicalBERT is optimized for clinical text (such as electronic medical records) and is suitable for analyzing patient-related clinical data.


## 📝 Input Summary
### 📝 User Inputs:

#### 🩺 Symptoms:
🔹 Is chest pain aggravated by exertion?: No
🔹 Is it a pressing or tightening sensation?: No
🔹 Does it last more than 5 minutes?: Yes
🔹 Does it radiate to shoulder/back/jaw?: No
🔹 Is it relieved by rest?: No
🔹 Is it accompanied by cold sweat?: Yes
🔹 Is there shortness of breath?: No
🔹 Is there nausea or vomiting?: No
🔹 Is there dizziness or fainting?: No
🔹 Is there palpitations?: No

#### 🏥 Medical History:
🔹 Do you have hypertension?: No
🔹 Do you have diabetes?: Yes
🔹 Do you have hyperlipidemia?: No
🔹 Do you smoke?: Yes
🔹 Family history of heart disease?: No
🔹 Recent emotional stress?: No

#### 🧪 Lab Parameters:
🔹 Systolic BP (mmHg): 120
🔹 Diastolic BP (mmHg): 80
🔹 LDL Cholesterol (mg/dL): 84.0
🔹 HDL Cholesterol (mg/dL): 76.0
🔹 Total Cholesterol (mg/dL): 185.0
🔹 Troponin I/T (ng/mL): 0.01
### File Content Analysis
{
  "LDL Cholesterol": "84 mg/dL (Ref: < 135 mg/dL)",
  "Total Cholesterol": "185 mg/dL (Ref: < 200 mg/dL)",
  "HDL Cholesterol": "76 mg/dL (Ref: ≥ 40 mg/dL)",
  "Non-HDL Cholesterol": "109 mg/dL (Ref: < 162 mg/dL)",
  "Triglycerides": "144 mg/dL (Ref: < 150 mg/dL)",
  "A1c": "5.4% (Ref: < 6.0%)",
  "eGFR": "72 mL/min/1.73m² (Ref: ≥ 60)",
  "Urea (BUN equivalent)": "23 mg/dL (Ref: ~7 – 23 mg/dL)",
  "Iron": "67 µg/dL (Ref: 40 – 160 µg/dL)",
  "Vitamin B12": "149 pg/mL (Ref: 148–220: Insufficiency)",
  "PSA (Prostate Specific Antigen)": "1.68 ng/mL (Ref: < 3.5 ng/mL)",
  "DHEAS": "148 µg/dL (Ref: ~69 – 305 µg/dL)",
  "WBC Count": "4.1 x10⁹/L (Ref: 4.5 – 11.0 x10⁹/L)",
  "RBC Count": "4.9 x10¹²/L (Ref: 4.4 – 5.9 x10¹²/L)",
  "Hemoglobin": "15.2 g/dL (Ref: 14.0 – 18.0 g/dL)",
  "Lymphocytes": "0.8 x10⁹/L (Ref: 1.0 – 3.3 x10⁹/L)",
  "Ferritin": "473 ng/mL (Ref: > 220 ng/mL)",
  "Platelets": "170 x10⁹/L (Ref: 140 – 440 x10⁹/L)"
}

Overriding lab parameter LDL Cholesterol (mg/dL):84.0 with original value 100
Overriding lab parameter Total Cholesterol (mg/dL):185.0 with original value 200
Overriding lab parameter HDL Cholesterol (mg/dL):76.0 with original value 50
"""
  
    language = "中文"  # or "English"
    # language = "English"  # or "中文"
    if language == "中文":
        model_outputs = model_outputs_chinese
    else:
        model_outputs = model_outputs_english
    print("=== Summary Result ===")
    summary = summarize_model_outputs_llm(model_outputs, language)
    print(summary)
