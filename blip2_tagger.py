import gradio as gr
from transformers import AutoProcessor,  Blip2ForConditionalGeneration
import torch
from PIL import Image


torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cats.jpg')
torch.hub.download_url_to_file('https://huggingface.co/datasets/nielsr/textcaps-sample/resolve/main/stop_sign.png',
                               'stop_sign.png')
torch.hub.download_url_to_file('https://cdn.openai.com/dall-e-2/demos/text2im/astronaut/horse/photo/0.jpg',
                               'astronaut.jpg')



blip2_processor_8_bit = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b")
blip2_model_8_bit = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b", device_map="auto",
                                                                  load_in_8bit=True)


device = "cuda" if torch.cuda.is_available() else "cpu"


# blip2_model.to(device)

def generate_caption(processor, model, image, tokenizer=None, use_float_16=False):
    inputs = processor(images=image, return_tensors="pt").to(device)

    if use_float_16:
        inputs = inputs.to(torch.float16)

    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)

    if tokenizer is not None:
        generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_caption



def generate_captions(image):

    caption_blip2_8_bit = generate_caption(blip2_processor_8_bit, blip2_model_8_bit, image, use_float_16=True).strip()

    return caption_blip2_8_bit


if __name__ == '__main__':
    image = Image.open("sample.jpg")

    print("D")