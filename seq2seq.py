import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader


class CaptionTagDataset(Dataset):
    def __init__(self, captions, tag_strs, tokenizer, max_len=128):
        self.captions = captions
        self.tag_strs = tag_strs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        tag_str = self.tag_strs[index]

        inputs = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        labels = self.tokenizer.encode(
            tag_str,
            add_special_tokens=False
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(labels)
        }


def train(model, tokenizer, train_dataset, val_dataset, epochs=5, batch_size=8, lr=1e-4):
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dataset) * epochs)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(epochs):
        running_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}')

        eval_loss = evaluate(model, tokenizer, val_loader, criterion)
        print(f'Epoch {epoch + 1}/{epochs}, Val Loss: {eval_loss:.4f}')


def evaluate(model, tokenizer, dataloader, criterion, device):
    model.eval()
    running_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = criterion(outputs.logits.view(-1, outputs.logits.shape[-1]), labels.view(-1))
            running_loss += loss.item()

    return running_loss / len(dataloader)


def predict(model, tokenizer, input_str):
    """
    input_str = "a cat is sleeping on a bed"
    predicted_tags = predict(model, tokenizer, input_str)
    print(predicted_tags)

    """
    model.eval()
    inputs = tokenizer.encode_plus(
        input_str,
        add_special_tokens=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    predicted_tags = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_tags


if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('sample.csv')

    # Initialize the tokenizer and the model
    tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=128)
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Create the datasets
    train_dataset = CaptionTagDataset(df['caption'][:8000], df['tag_str'][:8000], tokenizer)
    val_dataset = CaptionTagDataset(df['caption'][8000:], df['tag_str'][8000:], tokenizer)

    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train(model, tokenizer, train_dataset, val_dataset, epochs=5, batch_size=8, lr=1e-4)