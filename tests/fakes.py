"""Test doubles (tests only). FakeMedGemma mimics the server-side model interface."""
import json


class FakeMedGemma:
    model_id = "fake-medgemma"
    has_adapter = False

    def __init__(self, image_text: str = "", llm_reply: str | None = None):
        self.image_text, self.calls = image_text, []
        self.llm_reply = llm_reply or (
            "### Differential diagnosis\n1. Community-acquired pneumonia - high - fever, cough, RLZ opacity\n"
            "2. Acute bronchitis - moderate - cough\n3. Pulmonary TB - low - if cough > 2 weeks\n"
            "### Key findings used\n- RLZ opacity\n### Suggested next investigations\n- CBC, CRP\n"
            "### Red flags / when to seek urgent care\n- SpO2 < 92%")

    def generate(self, system, user, image=None, use_adapter=True, max_new_tokens=512):
        self.calls.append({"system": system, "user": user, "image": image is not None, "use_adapter": use_adapter})
        if image is not None:
            return json.dumps({"modality": "chest x-ray", "findings": ["patchy opacity right lower zone"],
                               "impression": "consolidation", "visible_text": self.image_text})
        return self.llm_reply
