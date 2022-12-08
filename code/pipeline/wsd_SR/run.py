from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from nltk.corpus import wordnet as wn

from dataset import WordSenseDisambiguationDataset
from processor import Processor
from model import SimpleModel

from config import parameters as conf

processor = Processor.from_config(conf.processor_path)

# test_dataset = WordSenseDisambiguationDataset(conf.model_input) 

# test_dataloader = DataLoader(
#     test_dataset,
#     batch_size=args.batch_size,
#     num_workers=args.num_workers,
#     collate_fn=processor.collate_sentences)

model = SimpleModel.load_from_checkpoint(conf.model_path)
device = 'cuda' if torch.cuda.is_available() and conf.device == 'cuda' else 'cpu'
model.to(device)
model.eval()

predictions = {}

with torch.no_grad():
    for x, _ in test_dataloader:
        x = {k: v.to(device) if not isinstance(v, list) else v for k, v in x.items()}
        y = model(x)
        batch_predictions = processor.decode(x, y)
        predictions.update(batch_predictions)

predictions = sorted(list(predictions.items()), key=lambda kv: kv[0])
