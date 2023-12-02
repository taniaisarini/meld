import matplotlib.pyplot as plt
import csv

import torch
import pickle
import utils
from utils import *

if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# dev data
batch_size = 100
dev_file_csv = "data/dev_sent_emo.csv"
test_file_csv = "data/test_sent_emo.csv"
train_file_csv = "data/train_sent_emo.csv"

df_dev = pd.read_csv(dev_file_csv)
print("Dev dataset length: {}".format(df_dev.shape[0]))
df_train = pd.read_csv(train_file_csv)
print("Train dataset length: {}".format(df_train.shape[0]))
df_test = pd.read_csv(test_file_csv)
print("Test dataset length: {}".format(df_test.shape[0]))



def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.array(labels.flatten())
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# Dataset
train_dataset = utils.create_speaker_dataset(df_train)
val_dataset = utils.create_speaker_dataset(df_dev)
test_dataset = utils.create_speaker_dataset(df_test)

print("train: {} validation: {} test: {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))


train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
test_dataloader = DataLoader(
            test_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=7,
                                                      output_attentions=False,
                                                      output_hidden_states=False).to(device)

optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

def train_validate():
    epoch_list = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    epochs = 50
    avg_val_loss = 10
    for i in range(epochs):
        epoch_list.append(i + 1)

        total_train_loss = 0
        train_accuracy = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            model_return = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            loss = model_return.loss
            logits = model_return.logits
            train_accuracy += flat_accuracy(logits.detach().numpy(), b_labels)

            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_accuracy = train_accuracy / len(train_dataloader)

        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        print("  Epoch: ", i)
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Average training accuracy: {0:.2f}".format(avg_train_accuracy))

        # Eval
        total_eval_accuracy = 0
        total_eval_loss = 0

        model.eval()

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                model_eval = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

                # print(model_eval)
                loss = model_eval.loss
                logits = model_eval.logits

                total_eval_loss += loss
                total_eval_accuracy += flat_accuracy(logits.detach().numpy(), b_labels)

        val_loss_diff = avg_val_loss
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        val_loss_diff = val_loss_diff - avg_val_loss
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        print("  Average validation loss: {0:.2f}".format(avg_val_loss))
        print("  Average validation accuracy: {0:.2f}".format(avg_val_accuracy))

        if val_loss_diff < 0.001:
            print("Early stopping ")
            break

    return (epoch_list, train_losses, train_accuracies, val_losses, val_accuracies)


def test():
    accuracy = 0
    with open('ouputs/speaker/output1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        model.eval
        for batch in test_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                model_eval = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

                accuracy += flat_accuracy(model_eval.logits.detach().numpy(), b_labels)
                pred_flat = np.argmax(model_eval.logits, axis=1).flatten()
                labels_flat = np.array(b_labels.flatten())

                for i in range(pred_flat.shape[0]):
                    prediction = label_to_emotion(pred_flat[i].item())
                    true = label_to_emotion(labels_flat[i])
                    writer.writerow((prediction, true))

    print("Test accuracy: {}%".format(accuracy/len(test_dataloader) * 100))


def function():
    # Train
    epoch_list, train_losses, train_accuracies, val_losses, val_accuracies = train_validate()

    with open('speaker_model_pkl', 'wb') as files:
        pickle.dump(model, files)

    # Loss plot
    plt.plot(epoch_list, train_losses, linestyle='-', marker='o', color='r')
    plt.plot(epoch_list, val_losses, linestyle='-', marker='x', color='g')
    plt.savefig("loss.png")
    plt.clf()

    # Accuracy plot
    plt.plot(epoch_list, train_accuracies, linestyle='-', marker='o', color='r')
    plt.plot(epoch_list, val_accuracies, linestyle='-', marker='x', color='g')
    plt.savefig("accuracy.png")

    # Test
    test()


function()
