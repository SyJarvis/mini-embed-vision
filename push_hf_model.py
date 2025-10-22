from transformers import AutoTokenizer, AutoImageProcessor
from transformers import AutoModel

def push_to_hf():
    model_dir = "./mini-embed-vision-v1.0"
    def init_model():
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        return model, image_processor, tokenizer

    model, image_processor, tokenizer = init_model()
    # 推送到huggingface
    model.push_to_hub("mini-embed-vision-v1.0")
    image_processor.push_to_hub("mini-embed-vision-v1.0")
    tokenizer.push_to_hub("mini-embed-vision-v1.0")
    # tokenizer.push_to_hub("minimind-v1-small", safe_serialization=False)

if __name__ == '__main__':
    push_to_hf()