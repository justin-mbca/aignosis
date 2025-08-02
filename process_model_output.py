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
    def format_model_outputs(outputs):
        formatted = ""
        for item in outputs:
            probs = "\n    ".join([f"🔹 {k}: {v:.2f}" for k, v in item["probabilities"].items()])
            formatted += (
                f"- **{item['model_name']}**:\n"
                f"  **风险等级:** {item['most_likely']}\n"
                f"  **概率分布:**\n"
                f"    {probs}\n"
                f"  **模型解释:** {item['explanation']}\n\n"
            )
        return formatted

    formatted_text = format_model_outputs(model_outputs)

    if language == "中文":
        prompt = f"""
你是一个医学风险分析助手。

请根据以下三个生物医学语言模型（BioBERT、PubMedBERT、ClinicalBERT）对某文本的风险预测，进行总体分析与判断。

每个模型提供：
- 风险等级（高风险、中风险、低风险）
- 概率分布
- 模型的简要解释

模型输出如下：

{formatted_text}

任务：
1. 总结三个模型对该文本的整体风险等级判断。
2. 分析模型之间的一致性或差异。
3. 给出系统是否应将该文本分类为“高风险”、“中风险”或“低风险”的建议。
4. 请额外添加 "建议用户行动" 字段，为非专业用户提供建议，解释是否需要看医生、是否紧急、是否可以等待观察、以及应准备哪些信息。

如果可能，请在英文翻译前加上"[English Translation]"，以便国际团队识别。
"""
    else:
        prompt = f"""
You are a medical risk analysis assistant.

Given the risk assessment outputs from three different biomedical language models, analyze and summarize the overall risk based on their classifications and probability distributions.

Each model provides:
- a risk level (e.g., High Risk, Medium Risk, Low Risk)
- a probability for the assigned level
- a short explanation of the model's training background

Here are the model outputs:

{formatted_text}

Task:
1. Summarize the consensus risk level from the three models.
2. Explain any disagreement between models.
3. Suggest whether this case should be treated as "High Risk", "Medium Risk", or "Low Risk" in an automated system.
4. Please also add a "User Action Suggestion" field to provide non-expert users with advice on whether they need to see a doctor, if it's urgent, if they can wait and observe, and what information they should prepare.
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
    language = "中文"  # or "English"
    model_outputs = [
        {
            "model_name": "BioBERT",
            "most_likely": "高风险",
            "probabilities": {"高风险": 0.39756593108177185},
            "explanation": "BioBERT 是一个专门针对生物医学文本训练的模型，适用于分析医学相关的文本。"
        },
        {
            "model_name": "PubMedBERT",
            "most_likely": "低风险",
            "probabilities": {"低风险": 0.5041127800941467},
            "explanation": "PubMedBERT 是基于 PubMed 数据训练的模型，专注于生物医学文献的理解。"
        },
        {
            "model_name": "ClinicalBERT",
            "most_likely": "高风险",
            "probabilities": {"高风险": 0.4117318391799927},
            "explanation": "ClinicalBERT 是针对临床文本（如电子病历）优化的模型，适合分析患者相关的临床数据。"
        }
    ]
    print("=== Summary Result ===")
    summary = summarize_model_outputs_llm(model_outputs, language)
    print(summary)
