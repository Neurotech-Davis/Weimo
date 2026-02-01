from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import traceback

DEBUG = False

class VLModel():

    def __init__(self, model_name="HuggingFaceTB/SmolVLM-Instruct", processor_name="HuggingFaceTB/SmolVLM-Instruct"):

        if DEBUG:
            self.device = None
            self.processor = None
            self.model = None
        else:
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Using device: {self.device}")
                print(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
                
                self.load_VLM_hf(model_name=model_name, processor_name=processor_name)
            except Exception as e:
                print(f"Error initializing VLModel: {str(e)}")
                traceback.print_exc()
                raise

    def load_VLM_hf(self, processor_name, model_name):
        try:
            print(f"Loading processor from {processor_name}...")
            self.processor = AutoProcessor.from_pretrained(processor_name)
            
            print(f"Loading model from {model_name}...")
            self.model = AutoModelForImageTextToText.from_pretrained(model_name,
                                                            dtype=torch.bfloat16,
                                                            _attn_implementation="flash_attention_2" if self.device == "cuda" else "eager").to(self.device)
                        
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            traceback.print_exc()
            raise

    def format_input(self, system_prompt="", message_prompt="", image=None):

        # Default inputs
        if system_prompt == "":
            system_prompt = "You are a VLM that is designed to detect obstacles to ensure a safe route for wheelchair users."
        if message_prompt == "":
            message_prompt = "Going forward, are there any obstacles to be aware of?"
        if image is None:
            print("Error: No image received")
            return None, None

        image1 = image

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
                    {"type": "text", "text": message_prompt}
                ]
            },
        ]
        return image1, messages

    # Runs inference
    def inference(self, system_prompt="", message_prompt="", max_new_tokens=500, image=None):

        if DEBUG:
            return "It works"

        # Format input
        image1, messages = self.format_input(
            system_prompt=system_prompt,
            message_prompt=message_prompt,
            image=image
        )
        
        # Check if format_input failed
        if image1 is None or messages is None:
            return "Error: Invalid input"

        # Prepare inputs
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image1], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate outputs
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return generated_texts[0].split("\n")[-1].split('Assistant: ')[1] # Returns only the generated texts