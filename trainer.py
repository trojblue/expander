import torch
import torch.nn as nn
from transformers import (
    EncoderDecoderModel,
    EncoderDecoderConfig,
    TrainingArguments,
    Trainer,
)
from transformers.models.bart.modeling_bart import shift_tokens_right
import numpy as np

from model.tokenizer import TagTokenizer

# Define the configuration of the encoder and decoder
config = EncoderDecoderConfig.from_encoder_decoder_configs(
    encoder_config=EncoderConfig(
        ...
    ),  # Replace ... with the desired encoder configuration
    decoder_config=DecoderConfig(
        ...
    ),  # Replace ... with the desired decoder configuration
)

# Initialize the encoder-decoder model
model = EncoderDecoderModel(config=config)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Replace ./results with the desired output directory
    num_train_epochs=3,  # Replace 3 with the desired number of training epochs
    per_device_train_batch_size=16,  # Replace 16 with the desired batch size
    save_total_limit=2,  # Replace 2 with the desired number of checkpoints to save
    logging_steps=500,
    save_steps=1000,
)

# Define the function to prepare the data for training
def prepare_data(data):
    tokenizer = TagTokenizer()

    input_ids = []
    output_ids = []
    weights = []
    for tags, prob in data:
        tag_ids = [tokenizer.tag_to_id[tag] for tag in tags]
        input_ids.append(tag_ids)
        output_tags = tokenizer.expand_tags(tags)
        output_ids.append([tokenizer.tag_to_id[tag] for tag in output_tags])
        # Compute weights for each tag based on its position and probability
        tag_weights = np.zeros(len(tag_ids))
        for i, tag in enumerate(tags):
            tag_prob = prob.get(tag, 0)
            tag_weights[i] = (1 / (i + 1)) * tag_prob
        tag_weights /= np.sum(tag_weights)
        weights.append(tag_weights)
    return {
        "input_ids": input_ids,
        "decoder_input_ids": shift_tokens_right(output_ids, tokenizer.pad_token_id),
        "weights": weights,
    }


# Define the weighted cross-entropy loss function
def weighted_cross_entropy_loss(logits, labels, weights):
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    losses = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
    weights = weights.view(-1)
    weighted_losses = losses * weights
    return torch.mean(weighted_losses)


# Define the trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=prepare_data,
    compute_loss=lambda model, inputs, **kwargs: weighted_cross_entropy_loss(
        model(**inputs).logits, inputs["decoder_input_ids"], inputs["weights"]
    ),
)

trainer.train()
