import gradio as gr
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image
from pathlib import Path

import os
os.environ["HF_HOME"] = str(Path("D:/Andrew/Cache/Huggingface"))

blip2_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")


def generate_caption(processor, model, image, tokenizer=None, use_float_16=False):
    inputs = processor(images=image, return_tensors="pt").to(device)

    if use_float_16:
        inputs = inputs.to(torch.float16)

    # Convert input tensor to float
    inputs.pixel_values = inputs.pixel_values.float()

    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)

    if tokenizer is not None:
        generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_caption



def generate_captions(image):
    caption_blip2 = generate_caption(blip2_processor, blip2_model, image, use_float_16=True).strip()
    return caption_blip2


def check_CUDNN():
    import torch

    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CuDNN version: {torch.backends.cudnn.version()}")

        # Check if CuDNN is enabled and working
        x = torch.randn(1, 3, 224, 224).cuda()
        y = torch.randn(1, 3, 224, 224).cuda()
        z = torch.nn.functional.conv2d(x, y)
        print("CuDNN is working properly!")
    else:
        print("CUDA is not available!")


if __name__ == '__main__':
    check_CUDNN()
    # image = Image.open("sample.jpg")

    # generate_captions(image)

    print("Done")
