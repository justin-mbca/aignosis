import openai
import json
from docx import Document
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load .docx content
def load_docx_text(path):
    doc = Document(path)
    return "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])

# Prepare prompt
def generate_prompt(text):
    return f"""
You are a medical document analysis assistant.

Given the following health checkup document text:

\"\"\"{text}\"\"\"

Extract all meaningful medical key-value pairs, such as lab test names and their corresponding values, and return them in flat JSON format like:

{{
  "Test Name 1": "Value 1",
  "Test Name 2": "Value 2"
}}

Include units and reference ranges where applicable. Do not group by section.
"""

# Call OpenAI API
def extract_key_value_pairs(docx_path):
    text = load_docx_text(docx_path)
    prompt = generate_prompt(text)

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    reply = response.choices[0].message.content
    try:
        data = json.loads(reply)
    except json.JSONDecodeError:
        print("⚠️ Warning: Could not parse response as JSON. Here's the raw response:\n")
        print(reply)
        return None
    return data

# Example usage
if __name__ == "__main__":
    result = extract_key_value_pairs("血脂.docx")
    if result:
        print(json.dumps(result, indent=2, ensure_ascii=False))
