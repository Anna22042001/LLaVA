from llava.model.builder import load_pretrained_model


tokenizer, model, image_processor, context_len = load_pretrained_model("Lin-Chen/ShareGPT4V-7B", None, "llava-v1.5-7b", False, False)