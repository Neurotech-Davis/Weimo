from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
from transformers.image_utils import load_image

# Just a class
class VLM():

    # Initialize variables
    def __init__(self, model_name, processor_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_VLM_hf()

    # Loads VLM through HuggingFace
    def load_VLM_hf(self, processor_name="HuggingFaceTB/SmolVLM-Instruct", model_name="HuggingFaceTB/SmolVLM-Instruct"):
        self.processor = AutoProcessor.from_pretrained(processor_name)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name,
                                                        dtype=torch.bfloat16,
                                                        _attn_implementation="flash_attention_2" if self.device == "cuda" else "eager").to(self.device)

    # Preprocess model inputs
    def format_input(self, system_prompt="", message_prompt="", image_path=""):

        # Default inputs
        if system_prompt=="":
            system_prompt = "You are a VLM that is designed to detect obstacles to ensure a safe route for wheelchair users."
        if message_prompt=="":
            message_prompt = "Going forward, are there any obstacles to be aware of?"
        if image_path=="":
            print("Error: No image path specified")
            return None

        # Load image
        image1 = load_image(image_path)

        # Create input messages
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    # {"type": "image"}, # If you wanna add a 2nd image this is where to put it
                    {"type": "text", "text": message_prompt}
                ]
            },
        ]
        return image1, messages

    # Runs inference
    def inference(self, system_prompt="", message_prompt="", image_path=""):

        # Format input
        image1, messages = self.format_input(system_prompt=system_prompt,
                                        message_prompt=message_prompt,
                                        image_path=image_path,
                                        processor=self.processor,
                                        model=self.model)

        # Prepare inputs
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image1], return_tensors="pt")
        inputs = inputs.to(self.device)

        # Generate outputs
        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return generated_texts[0]




