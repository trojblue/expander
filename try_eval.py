import torch
from transformers import AutoTokenizer
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration



def try_eval_1():
    # Load the saved model
    checkpoint = torch.load("./checkpoints/checkpoint_bench.pt", map_location=torch.device('cuda'))
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model.load_state_dict(checkpoint['model_state_dict'])


    # Load the saved model
    # model = T5ForConditionalGeneration.from_pretrained("./checkpoints/checkpoint_bench.pt")

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=512)

    while True:
        # Example input
        input_str = input("give me a input")

        # Tokenize the input
        inputs = tokenizer.encode_plus(
            input_str,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        # Perform inference
        model.eval()
        with torch.no_grad():
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            predicted_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(predicted_output)


def try_eval_2():
    import torch
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    # Load the trained checkpoint
    checkpoint_path = './checkpoints/checkpoint_2.pt'
    checkpoint = torch.load(checkpoint_path)

    # Load the tokenizer and the model
    tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model.load_state_dict(checkpoint['model_state_dict'])

    input_texts = [
        "A cute puppy sitting on the grass",
        "a woman in a black outfit sitting on a bench with her legs crossed and her legs crossed, with a building in the background",
        "a group of people in the water with a boat and a bird flying above them and a woman in a bikini",
        "1girl",
        "1girl, long hair"
    ]
    outputs = []
    for input_text in input_texts:

        # Prepare the input text
        input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True)

        # Generate the output text
        output_ids = model.generate(input_ids)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        outputs.append(output_text)

        # Print the output text
        print(output_text)

    print(outputs)



if __name__ == '__main__':
    try_eval_2()