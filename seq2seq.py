import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


class CaptionTagDataset(Dataset):
    def __init__(self, captions, tag_strs, tokenizer, max_len=128):
        self.captions = captions
        self.tag_strs = tag_strs
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Find the maximum length of all tag strings in the dataset
        self.max_tag_len = max(len(self.tokenizer.encode(tag_str)) for tag_str in self.tag_strs)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        tag_str = self.tag_strs[index]

        inputs = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,  # truncates inputs that exceed max_len
            max_length=self.max_len,
            return_tensors='pt'
        )

        labels = self.tokenizer.encode(
            tag_str,
            add_special_tokens=False
        )

        # Truncate the label tensor to the maximum length
        labels = labels[:self.max_tag_len]

        # Pad the label tensor with the padding token to the maximum length
        padding_length = self.max_tag_len - len(labels)
        labels += [self.tokenizer.pad_token_id] * padding_length

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(labels)
        }


def train(model, tokenizer, train_dataset, val_dataset, epochs=5, batch_size=8, lr=1e-4):
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataset) * epochs
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=total_steps//10, T_mult=2, eta_min=lr/10)

    print(f"total steps: {total_steps} | loading data.....")
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    writer = SummaryWriter()

    model.train()
    for epoch in range(epochs):
        running_loss = 0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
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

                pbar.update(1)

            train_loss = running_loss / len(train_loader)
            print(f'Train Loss: {train_loss:.4f}')

            writer.add_scalar('Loss/Train', train_loss, epoch+1)

            eval_loss = evaluate(model, tokenizer, val_loader, criterion)
            print(f'Val Loss: {eval_loss:.4f}')

            writer.add_scalar('Loss/Val', eval_loss, epoch+1)

    writer.close()


def evaluate(model, tokenizer, dataloader, criterion):
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
    df = pd.read_csv('combined.csv')

    # Initialize the tokenizer and the model
    tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Create the datasets
    train_count = int(len(df['caption']) * 0.85)
    train_dataset = CaptionTagDataset(df['caption'][:train_count], df['tag_str'][:train_count], tokenizer)
    val_dataset = CaptionTagDataset(df['caption'][train_count:], df['tag_str'][train_count:], tokenizer)

    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train(model, tokenizer, train_dataset, val_dataset, epochs=5, batch_size=4, lr=1e-4)