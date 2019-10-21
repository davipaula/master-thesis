from han_davi.data import HANData
from han_davi.model import HANModel
from han_davi.utils import Utils as utils
import random
import torch
import numpy as np
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="train")
    args = parser.parse_args()

    if args.task == "train":
        execute_training()
    else:
        execute_evaluation()

def execute_training():
    han_data = HANData()
    han_model = HANModel(han_data)

    training_steps(han_model)


def execute_evaluation():
    han_data = HANData()
    han_model = HANModel(han_data)

    evaluation_steps(han_model)


def training_steps(han_model):

    num_epoch = 1
    batch_size = han_model.batch_size

    x_train = han_model.X_train_pad
    y_train = han_model.y_train_tensor
    x_val = han_model.X_val_pad
    y_val = han_model.y_val_tensor
    sentence_attention_optimizer = han_model.sentence_optimizer
    sentence_attention_model = han_model.sentence_attention
    loss_criterion = han_model.criterion

    print_loss_every = 50
    code_test = True

    start = time.time()
    loss_full = []
    acc_full = []
    val_acc = []
    train_length = len(x_train)
    for i in range(1, num_epoch + 1):
        loss_epoch = []
        acc_epoch = []
        for j in range(int(train_length / batch_size)):
            print(batch_size)

            review, targets = gen_batch(x_train, y_train, batch_size)

            # loss,acc = train_data(batch_size, x, y, sentence_attention_model, sentence_attention_optimiser, loss_criterion)
            # def train_data(batch_size, review, targets, sentence_attention_model, sent_optimizer, criterion):

            state_word = sentence_attention_model.init_hidden_word()
            state_sent = sentence_attention_model.init_hidden_sent()
            sentence_attention_optimizer.zero_grad()

            y_pred, state_sent = sentence_attention_model(review, state_sent, state_word)

            loss = loss_criterion(y_pred, torch.LongTensor(targets))

            max_index = y_pred.max(dim=1)[1]
            correct = (max_index == torch.LongTensor(targets)).sum()
            acc = float(correct) / batch_size

            loss.backward()
            sentence_attention_optimizer.step()

            loss = loss.data.item()

            loss_epoch.append(loss)
            acc_epoch.append(acc)
            if (code_test and j % int(print_loss_every / batch_size) == 0):
                print('Loss at %d paragraphs, %d epoch,(%s) is %f' % (
                    j * batch_size, i, utils.timeSince(start), np.mean(loss_epoch)))
                print('Accuracy at %d paragraphs, %d epoch,(%s) is %f' % (
                    j * batch_size, i, utils.timeSince(start), np.mean(acc_epoch)))

        loss_full.append(np.mean(loss_epoch))
        acc_full.append(np.mean(acc_epoch))
        torch.save(sentence_attention_model.state_dict(), 'sentence_attention_model_yelp.pth')
        print('Loss after %d epoch,(%s) is %f' % (i, utils.timeSince(start), np.mean(loss_epoch)))
        print('Train Accuracy after %d epoch,(%s) is %f' % (i, utils.timeSince(start), np.mean(acc_epoch)))

        val_acc.append(utils.validation_accuracy(batch_size, x_val, y_val, sentence_attention_model))
        print('Validation Accuracy after %d epoch,(%s) is %f' % (i, utils.timeSince(start), val_acc[-1]))
        # return loss_full,acc_full,val_acc


def evaluation_steps(han_model):

    X_test_pad = han_model.X_test_pad
    y_test_tensor = han_model.y_test_tensor
    sentence_attention = han_model.sentence_attention
    batch_size = han_model.batch_size

    test_accuracy(batch_size, X_test_pad, y_test_tensor, sentence_attention)

    # Predict only a single batch
    x, y = utils.gen_batch(X_test_pad, y_test_tensor, batch_size)

    state_word = sentence_attention.init_hidden_word()
    state_sent = sentence_attention.init_hidden_sent()

    y_pred, state_sent = sentence_attention(x, state_sent, state_word)
    print(y_pred)  # probability per label

    max_index = y_pred.max(dim=1)[1]  # use label with highest probability as prediction
    print(max_index)

    print(y)  # true labels

    print(y_pred.shape)  # torch.Size([batch_size=8, labels=14])


def test_accuracy(batch_size, x_test, y_test, sentence_attention_model):
    acc = []
    test_length = len(x_test)
    for j in range(int(test_length / batch_size)):
        print(j)

        x, y = utils.gen_batch(x_test, y_test, batch_size)
        state_word = sentence_attention_model.init_hidden_word()
        state_sent = sentence_attention_model.init_hidden_sent()

        y_pred, state_sent = sentence_attention_model(x, state_sent, state_word)
        max_index = y_pred.max(dim=1)[1]
        correct = (max_index == torch.LongTensor(y)).sum()
        acc.append(float(correct) / batch_size)
    return np.mean(acc)


def gen_batch(x, y, batch_size):
    k = random.sample(range(len(x) - 1), batch_size)
    x_batch = []
    y_batch = []

    for t in k:
        x_batch.append(x[t])
        y_batch.append(y[t])

    return [x_batch, y_batch]


if __name__ == '__main__':
    main()
