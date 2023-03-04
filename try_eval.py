import torch
from transformers import AutoTokenizer
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration


if __name__ == '__main__':
    # Load the saved model
    checkpoint = torch.load("./checkpoints/checkpoint_bench.pt", map_location=torch.device('cpu'))
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model.load_state_dict(checkpoint['model_state_dict'])


    # Load the saved model
    # model = T5ForConditionalGeneration.from_pretrained("./checkpoints/checkpoint_bench.pt")

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

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