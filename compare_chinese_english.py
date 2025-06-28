import unittest, gradio
from dual_language import assess_with_huggingface

class TestDualLanguage(unittest.TestCase):
    def setUp(self):
        # Common inputs for both Chinese and English tabs
        self.structured_inputs = [
            "否", "否", "否", "否", "否", "否", "否", "否", "否", "否",  # Symptoms (Chinese)
            "否", "否", "否", "否", "否", "否",  # History (Chinese)
            120, 80, 140, 50, 250, 0.01  # Lab Parameters
        ]
        self.free_text_input = ""  # Empty free text input

    def test_chinese_and_english_alignment(self):
        # Run the Chinese tab analysis
        chinese_result = assess_with_huggingface(
            "中文",  # Language
            *self.structured_inputs,
            self.free_text_input
        )

        # Run the English tab analysis
        english_inputs = [
            "No", "No", "No", "No", "No", "No", "No", "No", "No", "No",  # Symptoms (English)
            "No", "No", "No", "No", "No", "No",  # History (English)
            120, 80, 140, 50, 250, 0.01  # Lab Parameters
        ]
        english_result = assess_with_huggingface(
            "English",  # Language
            *english_inputs,
            self.free_text_input
        )

        # Debug: Print results for manual verification
        print("Chinese Result:")
        print(chinese_result)
        print("\nEnglish Result:")
        print(english_result)

        # Assert that the results are aligned (ignoring language-specific differences)
        self.assertIn("低风险", chinese_result)  # Check for "Low Risk" in Chinese
        self.assertIn("Low Risk", english_result)  # Check for "Low Risk" in English

        self.assertIn("高脂血症", chinese_result)  # Check for "Hyperlipidemia" in Chinese
        self.assertIn("Hyperlipidemia", english_result)  # Check for "Hyperlipidemia" in English

        # Ensure the structure of the results is consistent
        self.assertEqual(len(chinese_result.splitlines()), len(english_result.splitlines()))

if __name__ == "__main__":
    unittest.main()