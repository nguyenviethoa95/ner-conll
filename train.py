from model import BiLSTM
import torch
import torch.nn as nn
from evaluation import accuracy, f1_macro
from preprocessing import prepare_sentence, create_tags_idx, load_pretrained_embedding, load_data
import argparse
import os


def train(device,
          embedding_dim,
          hidden_size,
          output_dim,
          num_layers,
          num_epochs,
          learning_rate,
          batch_size,
          embedding_path,
          training_data,
          validation_data,
          model_path):

    # LOAD THE TRAINING AND VALIDATION DATA
    train_sentences = load_data(training_data, zeros=True)
    val_sentences = load_data(validation_data, zeros=True)

    # PREPARE THE WORD EMBEDDING
    word2idx, weights = load_pretrained_embedding(embedding_path)
    tag2idx = create_tags_idx()

    # 1. INSTANTIATE THE MODEL CLASS
    model = BiLSTM(embedding_dim=embedding_dim,
                   hidden_dim=hidden_size,
                   output_dim=output_dim,
                   num_layers=num_layers,
                   weights=weights)
    model.to(device)
    print("Training on: " + str(device))

    # 2. INSTANTIATE THE LOSS CLASS
    normed_weights = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32).to(device)
    #     normed_weights = torch.tensor([1661.0, 835.0, 1668.0, 257.0, 1617.0, 1156.0, 702.0, 213.0,38323.0], dtype=torch.float32)
    #     normed_weights = (normed_weights/464.35)/normed_weights.sum()
    #     normed_weights = 1.0 / normed_weights
    #     normed_weights = normed_weights / normed_weights.sum()
    #     normed_weights = normed_weights.to(device, dtype = torch.float
    criterion = nn.CrossEntropyLoss(weight=normed_weights)

    # 3. INSTANTIATE THE OPTIMIZER CLASS
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 4. TRAIN THE MODEL
    for epoch in range(num_epochs):
        train_epoch_acc = 0
        train_epoch_loss = 0
        train_num_batches = train_sentences.shape[0]

        for batch_idx, samples in enumerate(train_sentences.values.tolist()):
            words = samples[1]
            tags = samples[2]

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Turn the tokens and tags into tensors of indexes
            indexed_sentence = prepare_sentence(words, word2idx)
            indexed_sentence = indexed_sentence.to(device)
            indexed_tags = prepare_sentence(tags, tag2idx)

            # Forward pass
            preds, __ = model(indexed_sentence, device)

            # Flatten the tags
            preds = preds.view(-1, preds.shape[-1])
            indexed_tags = indexed_tags.view(-1).to(device)

            # Calculate the loss
            batch_loss = criterion(preds, indexed_tags)

            # Getting gradients w.r.t. parameters
            batch_loss.backward()

            # Updating parameters
            optimizer.step()

            batch_acc = accuracy(preds.cpu(), indexed_tags.cpu())
            train_epoch_acc += batch_acc.item()
            train_epoch_loss += batch_loss.item()

        # 5. CALCULATE THE LOSS AND THE F1 SCORE ON THE VALIDATION DATASET
        val_epoch_loss = 0
        val_epoch_acc = 0
        val_f1_score = 0

        with torch.no_grad():
            val_num_batches = val_sentences.shape[0]
            for batch_idx, samples in enumerate(val_sentences.values.tolist()):
                words = samples[1]
                tags = samples[2]

                # Turn the tokens and tags into tensors of indexes
                indexed_sentence = prepare_sentence(words, word2idx)
                indexed_sentence = indexed_sentence.to(device)
                indexed_tags = prepare_sentence(tags, tag2idx)

                preds, _ = model(indexed_sentence, device)

                # Flatten the tags
                preds = preds.view(-1, preds.shape[-1])
                indexed_tags = indexed_tags.view(-1).to(device)

                # Calculate the loss and accuracy
                batch_acc = accuracy(preds.cpu(), indexed_tags.cpu())
                val_epoch_acc += batch_acc.item()
                batch_loss = criterion(preds, indexed_tags)

                preds = preds.detach().cpu().numpy()
                indexed_tags = indexed_tags.detach().cpu().numpy()
                val_f1_score += f1_macro(preds.argmax(axis=1), indexed_tags)

        print("Epoch: " + str(epoch))
        print(f"\tTrn Loss: {(train_epoch_loss / train_num_batches):.3f} "
              f"| Trn Acc: {(train_epoch_acc / train_num_batches) * 100:.2f}%")
        print(
            f"\tVal Loss: {(val_epoch_loss / val_num_batches):.3f} "
            f"| Val F1-Score: {(val_f1_score / val_num_batches):.3f} "
            f"|Val Acc: {(val_epoch_acc / val_num_batches) * 100:.2f}%")

    # 6. SAVE THE MODEL FILE
    torch.save(model.state_dict(), model_path+".ckpt")


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

    parser.add_argument('--embedding-path',
                        type=str,
                        default="data/glove.6B.50d/glove.6B.50d.txt",
                        help='path to the pretrained embedding file')

    parser.add_argument('--output_dim',
                        type=int,
                        default=9,
                        help='numbers of classes for the labels (default: 9)')

    parser.add_argument('--batch-size',
                        type=int,
                        default=1,
                        help='input batch size for training (default: 1)')

    parser.add_argument('--learning-rate',
                        type=float,
                        default=2e-05,
                        help='learning rate (default: 2e-05)')

    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='number of epochs to train (default: 20)')

    parser.add_argument('--gpu',
                        type=str,
                        default='cuda:2',
                        help='gpu (default: cuda:2)')

    parser.add_argument('--num-layers',
                        type=int,
                        default=1,
                        help='number of lstm layers (default: 1)')

    # Model files parameters
    parser.add_argument('--model-dir',
                        type=str,
                        help='name of the folder where models are stored.')
    parser.add_argument('--model-name',
                        type=str,
                        help='name of the stored trained model.')

    # Training data parameters
    parser.add_argument('--training-data-dir',
                        type=str,
                        help='name of the folder where training data is stored.')
    parser.add_argument('--training-data-file',
                        type=str,
                        help='path of the file where the training are stored.')
    parser.add_argument('--validation-data-dir',
                        type=str,
                        help='path of the file where the training are stored.')
    parser.add_argument('--validation-data-file',
                        type=str,
                        help='path of the file where the validation are stored.')

    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, args.model_name)
    training_data = os.path.join(args.training_data_dir, args.training_data_file)
    validation_data = os.path.join(args.validation_data_dir, args.validation_data_file)
    device = torch.device(args.gpu) if torch.cuda.is_available() else torch.device('cpu')

    train(device=device,
          embedding_dim=args.embedding_dim,
          hidden_size=args.hidden_size,
          output_dim=args.output_dim,
          num_layers=args.num_layers,
          num_epochs=args.epochs,
          learning_rate=args.learning_rate,
          batch_size=args.batch_size,
          embedding_path=args.embedding_path,
          training_data=training_data,
          validation_data=validation_data,
          model_path=model_path)
