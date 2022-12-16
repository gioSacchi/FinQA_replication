from argparse import ArgumentParser

import torch
from nltk.corpus import wordnet as wn

from torch.utils.data import DataLoader

from dataset import WordSenseDisambiguationDataset
from processor import Processor
from model import SimpleModel

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''

if __name__ == '__main__':
    parser = ArgumentParser()

    # Add data args.
    parser.add_argument('--processor', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_input', type=str, required=True)
    parser.add_argument('--model_output', type=str, required=True)

    # Add dataloader args.
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    # Other
    parser.add_argument('--device', type=str, default='cuda')

    # Store the arguments in hparams.
    args = parser.parse_args()

    processor = Processor.from_config(args.processor)

        ########

    import nltk
    from nltk import word_tokenize, pos_tag
    from nltk.stem import WordNetLemmatizer
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    # # nltk.download('tagsets')
    # nltk.download('universal_tagset')   

    lemmatizer = WordNetLemmatizer()
    sentence ="We were at the river bank and the river was flowing fast Then I went to a local bank to withdraw some money."
    words = word_tokenize(sentence)
    # words = [word.lower() for word in words]
    lemma_tokens = pos_tag(words)
    tags = [tag for _, tag in nltk.pos_tag(words, tagset='universal')]
    lemmas = []
    for word, tag in lemma_tokens:
        tag = get_wordnet_pos(tag)
        if tag != None and tag != "":
            lemmas.append(lemmatizer.lemmatize(word, tag))
        else:
            lemmas.append(lemmatizer.lemmatize(word))
    # lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in lemma_tokens]
    d = {"sentence_id": 0, "words": words, "lemmas": lemmas, "pos_tags": tags, "instance_ids":{5:"q1", 18:"q2", 20:"q3"}, "senses":{5:"q1", 18:"q2", 20:"q3"}}
    d1 = {"sentence_id": 0, "words": words, "lemmas": lemmas, "pos_tags": tags, "instance_ids":{5:"q11", 18:"q22", 22:"q33"}, "senses":{5:"q11", 18:"q22", 22:"q33"}}

    test_dataset = {0: d, 1: d1}

    ########

    # test_dataset = WordSenseDisambiguationDataset(args.model_input)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=processor.collate_sentences)

    model = SimpleModel.load_from_checkpoint(args.model)
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
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

    with open(args.model_output, 'w') as f:
        for instance_id, synset_id in predictions:
            f.write('{} {}\n'.format(instance_id, synset_id))
