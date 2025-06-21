from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Or another open LLM
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

def analyze_cardiovascular_text(freetext, lang="English"):
    if lang == "English":
        prompt = (
            "You are a medical assistant. Analyze the following patient description for cardiovascular risk factors, "
            "symptoms, and possible diseases. Give a concise summary and risk assessment.\n\n"
            f"Patient free text: {freetext}\n\n"
            "Analysis:"
        )
    else:
        prompt = (
            "你是一名医学助手。请分析以下患者描述，找出心血管风险因素、症状和可能的疾病，并给出简明总结和风险评估。\n\n"
            f"患者自由文本：{freetext}\n\n"
            "分析："
        )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result[len(prompt):].strip()