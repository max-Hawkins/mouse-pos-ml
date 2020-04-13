import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def load_data_from_csv(filename):
    data = pd.read_csv(filename, error_bad_lines=False, header=None).fillna(0).to_numpy(dtype='float')
    assert len(data[0]) % 2 == 0
    np.random.shuffle(data)
    data = np.reshape(data, (len(data), int(len(data[0]) / 2), 2))
    num_o = sum(label == 0 for label in data[:, 0, 0])
    num_x = sum(label == 1 for label in data[:, 0, 0])
    seq_length = len(data[0, :, 0])

    print('Raw Data Shape: ', np.shape(data))
    print('Total Data: ', len(data))
    print('Num O: ', num_o)
    print('Num X: ', num_x)
    print('Seq Length: ', seq_length)
    return data


class MousePosDataset(Dataset):
    def __init__(self, data):
        self.pos_data = data[:, 1:, :]
        self.labels = data[:, 0, 0]

    def __len__(self):
        return len(self.pos_data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.pos_data[idx]), self.labels[idx]


def split_train_test_val(all_data, batch_size, test_percent):

    test_data_size = (int)(test_percent * len(all_data))

    train_data = all_data[:-test_data_size]
    test_data = all_data[-test_data_size:]

    print('Num Train Data: ' + str(len(train_data)))
    print('Num Test Data:  ' + str(len(test_data)))

    train_dataset = MousePosDataset(train_data)
    test_dataset = MousePosDataset(test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dataloader, test_dataloader


def display_mouse_tour(data, label, predicted=None, subplot=False):

    title_color = 'grey'
    disp_text = 'Actual: ' + str(label)

    if predicted is not None:

        disp_text += f'  Predicted: {predicted:0.4f}'
        title_color = 'green' if int(round(predicted)) == label else 'red'

    # Scale plot boundaries to exclude the NaN fill value of 1.0
    x_min = min(data[:, 0])
    y_min = min(data[:, 1])
    x_max = max(d if d != 1.0 else 0.0 for d in data[:, 0]).item()
    y_max = max(d if d != 1.0 else 0.0 for d in data[:, 1]).item()

    colors = np.arange(len(data))
    if not subplot:
        plt.figure()
    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.scatter(data[:, 0], data[:, 1], c=colors)
    plt.title(disp_text, color=title_color, fontsize=15, fontweight='bold')
    if not subplot:
        plt.show()


def plot_loss_acc(args, train_loss, train_acc, test_loss, test_acc):
    plt.figure()
    plt.plot(train_loss, label='Training Loss', color='orange')
    plt.plot(test_loss, label='Test Loss', color='red')
    plt.plot(train_acc, label='Train Acc', color='blue')
    plt.plot(test_acc, label='Test Acc', color='green')
    plt.legend()
    plt.title(f'Batch: {args.batch_size:2.0f}, lr: {args.learning_rate:0.6f}, hlayers: {args.num_hidden_layers:3.0f}')
    plt.show()


def plot_conf_matrix(test_dataloader, model, device, batch_size):
    model.eval()
    all_categories = ['O', 'X']
    confusion = torch.zeros(2, 2)
    num_correct = 0
    num_wrong = 0
    subplot_dim = 4

    for data, labels in test_dataloader:
        seq = torch.FloatTensor(data)
        seq = np.swapaxes(seq, 0, 1)
        seq = seq.to(device)
        labels = labels.to(device)
        print('Input shape: ', np.shape(seq))
        model.hidden_cell = (torch.zeros(1, batch_size, model.hidden_layer_size, device=device),
                             torch.zeros(1, batch_size, model.hidden_layer_size, device=device))
        model_pred = model(seq)
        print('\nPredict: ' + str(model_pred))
        print('Actual: ' + str(labels))
        for i in range(batch_size):
            model_pred_int = int(model_pred[i].round())
            label = int(labels[i])

            num_correct += model_pred_int == label
            num_wrong += model_pred_int != label

            confusion[model_pred_int][label] += 1

            # display_mouse_tour(seq[:,i,:], label, model_pred[i].item())
            plt.subplot(subplot_dim, subplot_dim, i + 1)
            display_mouse_tour(seq[:, i, :].cpu(), label, model_pred[i].cpu().item(), True)

        plt.show()

    print(f'Num correct: {num_correct:3}  Num wrong: {num_wrong:3}')
    # Set up confusion matrix plot
    for i in range(2):
        confusion[i] = confusion[i] / confusion[i].sum()

    confusion_mat = pd.DataFrame(confusion.numpy(), index=all_categories, columns=all_categories)
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(confusion_mat, cmap='Greens', annot=True, annot_kws={"size": 16})  # font size
    plt.show()


# TODO: Display histogram of mouse tour lengths
def disp_data_len_hist(all_data):
    lengths = []

    for row in all_data:
        row = row[:, 0]
        row = row[~np.isnan(row)]
        length = len(row)

    print(length)
    lengths.append(length)

    sn.distplot(lengths)
    plt.show()


# TODO: Display hidden weights
def plot_hidden_weights(model):
    for param in model.named_parameters():
        if param[0] == 'lstm.weight_hh_l0':
            plt.imshow(param[1].detach().numpy())
            plt.show()


# Define LSTM model
class LSTM(nn.Module):
    # def __init__(self, input_size=2, hidden_layer_size=num_hidden_layers, output_size=batch_size):
    def __init__(self, input_size, hidden_layer_size, batch_size):
        super().__init__()
        device = 'cuda:0'
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, batch_size)
        # self.linear = nn.Linear(hidden_layer_size, output_size)

        self.sigmoid = nn.Sigmoid()

        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size, device=device),
                            torch.zeros(1, batch_size, self.hidden_layer_size, device=device))

    def init_hidden(self, batch_size):
        device = 'cuda:0'
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size, device=device),
                            torch.zeros(1, batch_size, self.hidden_layer_size, device=device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.reshape(len(input_seq), -1, 2), self.hidden_cell)
        linear_out = self.linear(lstm_out[-1])
        predictions = self.sigmoid(linear_out)

        return predictions[-1]


def train_epoch(train_dataloader, model, device, optimizer, scheduler, batch_size, loss_function, clip):
    model.train()
    train_losses = []
    train_acc = []

    for seq, labels in train_dataloader:

        seq = np.swapaxes(seq, 0, 1)
        seq = seq.to(device)
        labels = labels.to(device)

        # TODO: Implement class weights
        # weights = np.array(class_weights[label] for label in labels)
        # print('Weights: ' + str(weights))

        optimizer.zero_grad()
        model.init_hidden(batch_size)

        # TODO: Use packed/padded inputs
        # print(np.shape(seq))
        # packed = rnn_utils.pack_padded_sequence(padded, lengths)
        # packed_out, packed_hidden = lstm(packed)
        # unpacked, unpacked_len = rnn_utils.pad_packed_sequence(packed_out)

        y_pred = model(seq)

        train_loss = loss_function(y_pred, labels)
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        scheduler.step()

        train_losses.append(train_loss.item())
        train_acc.append(sum(y_pred.round() == labels).item() / len(labels))

    return np.average(train_losses), np.average(train_acc)


def test_epoch(test_dataloader, model, device, batch_size, loss_function):
    model.eval()  # TODO: Check if this is needed with torch.no_grad()
    test_losses = []
    test_acc = []
    with torch.no_grad():
        for seq, labels in test_dataloader:
            seq = np.swapaxes(seq, 0, 1)

            seq = seq.to(device)
            labels = labels.to(device)

            model.hidden_cell = (torch.zeros(1, batch_size, model.hidden_layer_size, device=device),
                                 torch.zeros(1, batch_size, model.hidden_layer_size, device=device))
            y_pred = model(seq)

            test_loss = loss_function(y_pred, labels)
            test_losses.append(test_loss.item())
            test_acc.append(sum(y_pred.round() == labels).item() / len(labels))
    return np.average(test_losses), np.average(test_acc)


def main():
    parser = argparse.ArgumentParser(description='ML Mouse Data Trainer')
    parser.add_argument('--batch-size', type=int, default=1, dest='batch_size',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=500, dest='epochs',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.001, dest='learning_rate',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--test_percent', type=float, default=0.2, dest='test_percent',
                        help='percentage of all data to use as test data (default: 0.2)')
    parser.add_argument('--num_hidden_layers', type=int, default=20, dest='num_hidden_layers',
                        help='number of hidden layers in LSTM (default: 20)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save', action='store_true', default=True,
                        help='save final trained model')
    # TODO: Implement saving and resuming
    # parser.add_argument('--save-interval', type=int, default=10, metavar='N',
    #                    help='how many batches to wait before checkpointing')
    # parser.add_argument('--resume', action='store_true', default=False,
    #                    help='resume training from checkpoint')

    args = parser.parse_args()
    print(args)

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('\nUsing device: ', torch.cuda.get_device_name())
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # Training Parameters
    csv_data_file = 'data/pos_data_excised.csv'
    save_model_path = 'models/ex_model.pt'
    test_percent = args.test_percent
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    num_hidden_layers = args.num_hidden_layers
    clip = 0.5

    # Load all data, split into train/test, then create dataloaders
    all_data = load_data_from_csv(csv_data_file)
    train_dataloader, test_dataloader = split_train_test_val(all_data, batch_size, test_percent)

    model = LSTM(2, num_hidden_layers, batch_size).to(device)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500], gamma=0.1)

    print(model)

    epoch_train_loss = []
    epoch_test_loss = []
    epoch_train_acc = []
    epoch_test_acc = []

    for i in range(epochs):
        train_loss, train_acc = train_epoch(train_dataloader, model, device, optimizer,
                                            scheduler, batch_size, loss_function, clip)
        test_loss, test_acc = test_epoch(test_dataloader, model, device, batch_size, loss_function)

        epoch_train_loss.append(train_loss)
        epoch_test_loss.append(test_loss)
        epoch_train_acc.append(train_acc)
        epoch_test_acc.append(test_acc)

        if i % 25 == 0 or i == epochs - 1:
            print(f'epoch: {i:3} Train loss: {epoch_train_loss[-1].item():10.8f}  Train Acc: {epoch_train_acc[-1].item():0.4f}')
            print(f'           Test loss: {epoch_test_loss[-1].item():10.8f}  Test Acc: {epoch_test_acc[-1].item():0.4f}')
            for param in model.named_parameters():
                if param[0] == 'lstm.weight_hh_l0':
                    print('Hidden Weights STD: ' + str(param[1].cpu().detach().numpy().std()))

    plot_loss_acc(args, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc)
    plot_conf_matrix(test_dataloader, model, device, batch_size)

    if args.save:
        torch.save(model.state_dict(), save_model_path)

if __name__ == '__main__':
    main()
