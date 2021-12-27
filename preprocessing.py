import codecs
import re
import pandas as pd
import torch


def load_data(fp, zeros):
    """
    Load the CONLL training data
    1. Separate each sentence, whilst
    2. Replace the digit with all zero
    3. Format the text data into the following form with
    id as sentence id, words is the tokens of the sentence,
    tags are the NER labels of the tokens.

    [["id": 0, "words": [Alex, is, going, to, Los, Angeles, in California] , "tags":[I-PER,O,O,O, B-LOC, I-LOC,O, I-LOC]],
    ["id": 1, "words": [Jim, bought, 300, shares, of, Acme, Corp], "tags":[I-PER,O,O,O,B-ORG, I-ORG]],
    .....
    ["id": n, "words": [...], "tags":[....]]]

    :param fp: filepath to the data
    :param zeros:
    :return:
    """
    sentences = []
    sentence = []

    for line in codecs.open(fp, "r", "utf-8"):
        line = line.rstrip() if zeros else line.rstrip()
        line = re.sub(r'\d', '0', line)  # Replace digit in a string with a zero
        if not line:
            if len(sentence) > 0:
                if "DOCSTART" not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)

    if len(sentence) > 0:
        if "DOCSTART" not in sentence[0][0]:
            sentences.append(sentence)

    sentences_df = pd.DataFrame(columns=["id", "words", "tags"])
    for i, sen in enumerate(sentences):
        words = []
        tags = []
        for word in sen:
            words.append(word[0].lower())
            tags.append(word[-1])
        sentences_df = sentences_df.append({"id": i, "words": words, "tags": tags}, ignore_index=True)

    return sentences_df


def prepare_sentence(sentence: list, word2idx: dict) -> list:
    """
    Map the tokens in a sentence into the corresponding index within the dictionary
    :param sentence: list of tokens
    :param word2idx: dict to map from token to the index in the vocabulary
    :return:
    """
    encoded_sentences = []
    for word in sentence:
        if word in word2idx.keys():
            encoded_sentences.append(word2idx[word])
        else:
            encoded_sentences.append(word2idx["<unk>"])
    return torch.LongTensor(encoded_sentences)


def load_pretrained_embedding(fp):
    """
    1. Load the pretrained Glove embeddings from file and add the <unk> token for
    the OOV (out-of-vocabulary) words.
    2. Create a look-up dictionary for mapping the token into the corresponding index in the vocabulary

    :param fp:
    :return:
    """
    embeddings_matrix = pd.read_csv(fp, sep=" ", header=None, quoting=3)

    # Inset the <unk> token at the first row
    data = ["<unk>"]
    unk_emb = list(embeddings_matrix.iloc[:, 1:].mean(axis=0).values)
    data += unk_emb
    embeddings_matrix.loc[-1] = data  # adding a row
    embeddings_matrix.index = embeddings_matrix.index + 1  # shifting index
    embeddings_matrix = embeddings_matrix.sort_index()  # sorting by index

    # Separate the list of tokens and the list of embedding from the dataframe
    vocab = embeddings_matrix[0]
    vectors = embeddings_matrix.iloc[:, 1:]
    weights = torch.FloatTensor(vectors.values.tolist())

    # Map words to unique indexes
    word2idx = {word: ind for ind, word in enumerate(vocab)}
    # idx2word = {ind: word for ind, word in enumerate(vocab)}
    word_pad_idx = 0
    return word2idx, weights


def create_tags_idx():
    tags = ["B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-MISC", "I-MISC", "O"]
    tag2idx = {t: i for i, t in enumerate(tags)}

    return tag2idx