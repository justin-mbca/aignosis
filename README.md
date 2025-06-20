## 🧠 Project Overview | 项目概述

**Aignosis** is a bilingual (English/中文) AI assistant that helps users assess their cardiovascular health risk based on structured symptoms and lab inputs. It bridges clinical logic with real-world accessibility using logic rules and optional GPT interpretation.

Aignosis 是一个双语心血管 AI 风险评估工具，结合结构化问卷与可选医学指标，提供早期风险分级与疾病预测。

---

## 🚀 Quick Start | 快速开始

1. Visit the Hugging Face Space → [点击此处访问](https://huggingface.co/spaces/zhangju2023/aignosis)
2. Choose your preferred language: English or 中文
3. Fill out key symptoms (e.g. chest pain, shortness of breath)
4. Enter optional health data: blood pressure, glucose, CK-MB, Troponin I
5. Submit to receive:
   - Personalized risk level
   - Possible condition prediction (e.g. Angina, MI)
   - Explanation in simple terms

---

## 🔍 Key Features | 核心功能

- 🌐 Bilingual UI: English and Simplified Chinese
- 🧠 Symptom-based logic mimicking HEART/Framingham risk scores
- 🧪 Supports cardiac markers: CK-MB and Troponin I
- ✍️ Free-text input + keyword interpretation (rule-based or GPT)
- 📃 Plain-language report with reasoning trace

---

## 📖 User Guide | 用户指南

| 步骤 | 操作说明 |
|------|-----------|
| ① | 选择语言（English 或 中文） |
| ② | 勾选或填写主要症状、病史与生活习惯 |
| ③ | 输入血压、血糖、心率等指标（可选） |
| ④ | （可选）添加自由描述，例如：“我在爬楼时胸闷” |
| ⑤ | 点击提交按钮，系统将提供风险评估结果与医学解释 |

---

## ⚙️ Technology Stack | 技术架构

- `Python 3.9+`
- `Gradio 5.34+`
- `Hugging Face Spaces`
- Optional NLP: `GPT (via OpenAI API)` / `spaCy` / `jieba`

---

## 📚 References | 临床参考

- Framingham Risk Score for chronic CVD prediction  
- HEART Score for acute chest pain triage  
- NICE Guidelines (UK): https://www.nice.org.uk/guidance/cg95

---

## 🗣 Feedback & Collaboration | 反馈与协作

We welcome contributors!  
Feel free to open Issues or Pull Requests on [GitHub](https://github.com/justin-mbca/aignosis) to improve logic, UI, translations, or add support for more biomarkers.

我们欢迎你在 GitHub 上提交建议、问题或功能更新，帮助我们共同打磨这个工具！
