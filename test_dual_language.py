import unittest
from dual_language import evaluate_cardiovascular_disease

class TestDiseaseClassification(unittest.TestCase):
    def test_myocardial_infarction(self):
        symptoms = {"Chest Pain": True, "Shortness of Breath": False}
        history = {}
        lab_params = {"Troponin I/T": 0.05}
        result = evaluate_cardiovascular_disease(symptoms, history, lab_params)
        self.assertIn("心肌梗塞 / Myocardial Infarction", result)

    def test_hyperlipidemia(self):
        symptoms = {}
        history = {}
        lab_params = {"Total Cholesterol": 250, "LDL-C": 140}
        result = evaluate_cardiovascular_disease(symptoms, history, lab_params)
        self.assertIn("高脂血症 / Hyperlipidemia", result)

    def test_heart_failure(self):
        symptoms = {"Shortness of Breath": True}
        history = {}
        lab_params = {"BNP": 150}
        result = evaluate_cardiovascular_disease(symptoms, history, lab_params)
        self.assertIn("心力衰竭 / Heart Failure", result)

    def test_no_disease(self):
        symptoms = {"Chest Pain": False, "Shortness of Breath": False}
        history = {}
        lab_params = {"Troponin I/T": 0.01, "Total Cholesterol": 180, "BNP": 50}
        result = evaluate_cardiovascular_disease(symptoms, history, lab_params)
        self.assertIn("无明显心血管疾病风险 / No significant cardiovascular disease risk detected", result)

if __name__ == "__main__":
    unittest.main()