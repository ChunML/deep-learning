from utils import read_labels_from_file, input_fn
from network import Network
import pandas as pd


def main():
    label_path = './data/all/labels.csv'
    batch_size = 64
    new_size = 224

    filenames, labels, int_to_breed, breed_to_int = read_labels_from_file(
        label_path)
    features, labels = input_fn(filenames, labels, batch_size)
    num_batches = len(filenames) // batch_size
    num_epochs = 100
    num_epochs_per_decay = 2
    decay_steps = num_epochs_per_decay * num_batches
    net = Network(features, labels, len(int_to_breed),
                  initial_learning_rate=0.01, decay_steps=decay_steps)
    train_info = net.train(num_epochs, num_batches)
    train_df = pd.DataFrame(train_info)
    train_df.to_csv('train_info_{}_epochs.csv'.format(num_epochs))


if __name__ == '__main__':
    main()
