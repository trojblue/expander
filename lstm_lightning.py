import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import wandb
import torch.nn.functional as F

class TagDataset(Dataset):
    def __init__(self, csv_file, max_length=50):
        self.data = pd.read_csv(csv_file)
        print("building vocab....")
        self.vocab = set(tag for tags in self.data["tag_str"] for tag in tags.split(","))
        self.tag_to_index = {tag: i for i, tag in enumerate(self.vocab)}
        self.index_to_tag = {i: tag for tag, i in self.tag_to_index.items()}
        self.max_length = max_length
        self.num_tags = len(self.vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tags = self.data.iloc[idx]["tag_str"].split(",")
        tags = [self.tag_to_index[tag] for tag in tags if tag in self.vocab]
        tags = tags[:self.max_length]  # truncate tags if longer than max_length
        tag_tensor = torch.zeros(self.max_length, dtype=torch.long)
        tag_tensor[:len(tags)] = torch.tensor(tags)
        return tag_tensor.unsqueeze(0)  # Add extra dimension

class LSTMTagger(pl.LightningModule):
    def __init__(self, num_tags, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.5, max_length = 64):
        super().__init__()
        self.num_tags = num_tags
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(num_tags, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tags),
            nn.Sigmoid()
        )

    def forward(self, input_tags):
        embedded = self.embedding(input_tags)
        output, _ = self.lstm(embedded)
        output = output[-1]  # Use only the last output of the LSTM
        output = self.mlp(output)
        return output

    def training_step(self, batch, batch_idx):
        input_tags = batch.squeeze(1)  # Remove extra dimension
        target_tags = input_tags.clone()
        target_tags[:, :-1] = input_tags[:, 1:]  # Shift tags by 1
        target_tags[:, -1] = 0  # Set last tag to padding index

        output_tags = self(input_tags)
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fn(output_tags.view(-1, self.num_tags), target_tags.view(-1))
        self.log('train_loss', loss)
        return loss



    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        return {"optimizer": optimizer, "scheduler": scheduler, "monitor": "train_loss"}

    def train_dataloader(self):
        dataset = TagDataset('tag_only.csv')
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        return dataloader

    def on_train_epoch_end(self):
        if self.current_epoch % 10 == 0:
            checkpoint_path = f"model_epoch_{self.current_epoch}.ckpt"
            self.logger.experiment.log_artifact(checkpoint_path)
            self.save_checkpoint(checkpoint_path)

if __name__ == '__main__':
    # wandb.init(project="tag-expander", entity="your-username")
    dataset = TagDataset('tag_only.csv')
    model = LSTMTagger(num_tags=dataset.num_tags, hidden_dim=256)
    # trainer = pl.Trainer(gpus=1, max_epochs=100, progress_bar_refresh_rate=20, logger=wandb_logger)
    trainer = pl.Trainer(gpus=1, max_epochs=100)

    trainer.fit(model)