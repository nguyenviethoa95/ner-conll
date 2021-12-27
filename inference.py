import torch
import argparse
import os

from model import BiLSTM
from preprocessing import load_pretrained_embedding, prepare_sentence, create_tags_idx, load_data
from evaluation import accuracy, f1_macro, _confusion_matrix, _report


def infer(model_path,
          device,
          embedding_dim,
          hidden_size,
          output_dim,
          num_layers,
          embedding_path,
          test_data):

    # 1. LOAD THE TEST DATA
    test_sentences = load_data(test_data, zeros=True)

    # 2. PREPARE THE WORD EMBEDDING
    word2idx, weights = load_pretrained_embedding(embedding_path)
    tag2idx = create_tags_idx()

    # Load the trained model
    model = BiLSTM(embedding_dim=embedding_dim,
                   hidden_dim=hidden_size,
                   output_dim=output_dim,
                   num_layers=num_layers,
                   weights=weights)

    model.load_state_dict(torch.load(model_path+".ckpt"))
    model.to(device)
    model.eval()

    predictions = []
    ground_truth = []

    with torch.no_grad():
        for samples in test_sentences.values.tolist():
            words = samples[1]
            tags = samples[2]

            # Turn the tokens and tags into tensors of indexes
            indexed_sentence = prepare_sentence(words, word2idx)
            indexed_sentence = indexed_sentence.to(device)
            indexed_tags = prepare_sentence(tags, tag2idx)

            preds, _ = model(indexed_sentence, device)

            # Flatten the tags
            preds = preds.view(-1, preds.shape[-1]).detach().cpu().numpy()
            indexed_tags = indexed_tags.view(-1).to(device).detach().cpu().numpy()

            predictions.extend(preds.argmax(axis=1))
            ground_truth.extend([idx for idx in indexed_tags])

    f1_macro_score = f1_macro(predictions, ground_truth)
    cm = _confusion_matrix(predictions, ground_truth)
    report = _report(predictions, ground_truth)
    print(f1_macro_score)
    print(cm)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--embedding-dim',
                        type=int,
                        default=50,
                        help='input batch size for training (default: 50)')

    parser.add_argument('--hidden-size',
                        type=int,
                        default=100,
                        help='lstm cell hidden size (default: 100)')

    parser.add_argument('--num-layers',
                        type=int,
                        default=1,
                        help='number of lstm layers (default: 1)')

    parser.add_argument('--embedding-path',
                        type=str,
                        default="data/glove.6B.50d/glove.6B.50d.txt",
                        help='path to the pretrained embedding file')

    parser.add_argument('--output_dim',
                        type=int,
                        default=9,
                        help='numbers of classes for the labels (default: 9)')

    parser.add_argument('--gpu',
                        type=str,
                        default='cuda:2',
                        help='gpu (default: cuda:2)')

    # Model files parameters
    parser.add_argument('--model-dir',
                        type=str,
                        help='name of the folder where models are stored.')
    parser.add_argument('--model-name',
                        type=str,
                        help='name of the stored trained model.')

    # Training data parameters
    parser.add_argument('--test-data-dir',
                        type=str,
                        help='name of the folder where training data is stored.')
    parser.add_argument('--test-data-file',
                        type=str,
                        help='path of the file where the training are stored.')

    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, args.model_name)
    test_data = os.path.join(args.test_data_dir, args.test_data_file)
    device = torch.device(args.gpu) if torch.cuda.is_available() else torch.device('cpu')

    infer(model_path=model_path,
          device=device,
          embedding_dim=args.embedding_dim,
          hidden_size=args.hidden_size,
          output_dim=args.output_dim,
          num_layers=args.num_layers,
          embedding_path=args.embedding_path,
          test_data=test_data)
