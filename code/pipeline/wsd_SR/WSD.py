from config import parameters as conf
import torch
import nltk
from nltk.corpus import wordnet as wn
from torch.utils.data import DataLoader
from processor import Processor
from model import SimpleModel
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

# download nltk packages first time
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# # nltk.download('tagsets')
# nltk.download('universal_tagset')   

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

def WSD(input):
    # load processor
    processor = Processor.from_config(conf.processor_path)

    lemmatizer = WordNetLemmatizer()

    # Create dataloader
    test_dataloader = DataLoader(
        input,
        batch_size=conf.batch_size,
        num_workers=conf.num_workers,
        collate_fn=processor.collate_sentences)

    # Load model
    model = SimpleModel.load_from_checkpoint(conf.model_path)
    device = 'cuda' if torch.cuda.is_available() and conf.device == 'cuda' else 'cpu'
    model.to(device)
    model.eval()

    # Run predictions
    predictions = {}
    with torch.no_grad():
        for x, _ in test_dataloader:
            x = {k: v.to(device) if not isinstance(v, list) else v for k, v in x.items()}
            y = model(x)
            batch_predictions = processor.decode(x, y)
            predictions.update(batch_predictions)

    predictions = sorted(list(predictions.items()), key=lambda kv: kv[0])

    # format predictions
    for instance_id, synset_id in predictions:
        predictions[instance_id] = synset_id

    # with open(conf.model_output, 'w') as f:
    #     for instance_id, synset_id in predictions:
    #         f.write('{} {}\n'.format(instance_id, synset_id))

    return predictions


# sentence ="We were at the river bank and the river was flowing fast Then I went to a local bank to withdraw some money."
# words = word_tokenize(sentence)
# # words = [word.lower() for word in words]
# lemma_tokens = pos_tag(words)
# tags = [tag for _, tag in nltk.pos_tag(words, tagset='universal')]
# lemmas = []
# for word, tag in lemma_tokens:
#     tag = get_wordnet_pos(tag)
#     if tag != None and tag != "":
#         lemmas.append(lemmatizer.lemmatize(word, tag))
#     else:
#         lemmas.append(lemmatizer.lemmatize(word))
# # lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in lemma_tokens]
# d = {"sentence_id": 0, "words": words, "lemmas": lemmas, "pos_tags": tags, "instance_ids":{5:"q1", 18:"q2", 20:"q3"}}
# d1 = {"sentence_id": 0, "words": words, "lemmas": lemmas, "pos_tags": tags, "instance_ids":{5:"q11", 18:"q22", 22:"q33"}}

# test_dataset = {0: d, 1: d1}
