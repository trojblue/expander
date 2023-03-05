import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup

import csv


class TagDataset(Dataset):
    def __init__(self, path, max_len):
        self.data = []
        self.tag_set = set()
        self.max_len = max_len
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                caption, tags = row
                tags_list = tags.split()
                caption = caption[:max_len]  # Truncate captions longer than max_len
                if len(caption) < max_len:  # Pad captions shorter than max_len
                    caption += ''.join([''] * (max_len - len(caption)))
                tags_list = tags_list[:max_len]  # Truncate tags longer than max_len
                if len(tags_list) < max_len:  # Pad tags shorter than max_len
                    tags_list += [''] * (max_len - len(tags_list))
                self.data.append((caption, tags_list))
                self.tag_set.update(tags_list)
        self.num_tags = len(self.tag_set)
        self.tag_list = list(self.tag_set) # Convert tag set to list

    def __getitem__(self, index):
        caption, tags = self.data[index]
        caption = torch.tensor([self.tokenizer.convert_tokens_to_ids(token) for token in caption])
        tags = torch.tensor([self.tag_list.index(tag) for tag in tags]) # use tag_list instead of tag_set
        return caption, tags

    def __len__(self):
        return len(self.data)


class TagExpansionModel(pl.LightningModule):
    def __init__(self, num_tags, hidden_dim):
        super().__init__()
        self.num_tags = num_tags
        self.encoder = nn.LSTM(num_tags, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(num_tags, hidden_dim, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tags)
        )

    def forward(self, captions, tags):
        # Encode the input tags
        _, (h_n, c_n) = self.encoder(tags)

        # Decode the input captions conditioned on the encoded tags
        _, (h_n, c_n) = self.decoder(captions, (h_n, c_n))

        # Generate the output tags from the final decoder hidden state
        output = self.mlp(h_n.squeeze())
        return output

    def training_step(self, batch, batch_idx):
        captions, tags = batch
        output = self(captions, tags)
        loss = F.binary_cross_entropy_with_logits(output, tags.float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        captions, tags = batch
        output = self(captions, tags)
        loss = F.binary_cross_entropy_with_logits(output, tags.float())
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    wandb.init(project='tag-expansion')
    # wandb_logger = pl.loggers.wandb.WandbLogger()
    wandb_logger = None
    max_len = 256
    train_data = TagDataset('tag_only.csv', max_len)
    val_data = TagDataset('tag_only_val.csv', max_len)
    num_tags = train_data.num_tags
    hidden_dim = 256
    model = TagExpansionModel(num_tags, hidden_dim)

    torch.set_float32_matmul_precision('high')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints/',
        filename='tag_expansion-{epoch:02d}-{val_loss:.2f}',
        save_top_k=-1,
        every_n_train_steps=10 * len(train_data)  # save every 10 epochs
    )


    trainer = pl.Trainer(gpus=1, logger=wandb_logger,
                         callbacks=[
                             checkpoint_callback
                            ]
                         )
    trainer.fit(model, DataLoader(train_data, batch_size=16), DataLoader(val_data, batch_size=16))