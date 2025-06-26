import torch
from torch import nn
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_iter = AG_NEWS(split='train')

tokenizer = get_tokenizer('basic_english')
def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return torch.tensor(label_list), text_list, torch.tensor(lengths)

train_iter = AG_NEWS(split='train')
dataloader = DataLoader(list(train_iter), batch_size=8, shuffle=True, collate_fn=collate_batch)

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, text, offsets=None):
        embedded = self.embedding(text)
        return self.fc(embedded)

VOCAB_SIZE = len(vocab)
EMBED_DIM = 64
NUM_CLASSES = 4
model = TextClassificationModel(VOCAB_SIZE, EMBED_DIM, 64, NUM_CLASSES).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    losses = []
    for label, text, lengths in dataloader:
        label, text = label.to(device), text.to(device)
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_acc += (output.argmax(1) == label).sum().item()
        total_count += label.size(0)
        losses.append(loss.item())
    return total_acc / total_count, sum(losses)/len(losses)

accs, losses = [], []
for epoch in range(5):
    acc, loss = train(dataloader)
    print(f"Epoch {epoch+1}: Accuracy={{acc:.2f}}, Loss={{loss:.4f}}")
    accs.append(acc)
    losses.append(loss)

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/text_model.pth")

os.makedirs("outputs", exist_ok=True)
plt.plot(accs, label='Accuracy')
plt.plot(losses, label='Loss')
plt.xlabel('Epoch')
plt.title("Accuracy and Loss Over Epochs")
plt.legend()
plt.savefig("outputs/loss_accuracy_plot.png")
plt.show()
