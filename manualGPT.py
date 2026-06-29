import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

# ---------------------------------------------------------
# 1. TOKENIZER + VOCAB
# ---------------------------------------------------------

def build_vocab(sentences):
    counter = Counter()
    for s in sentences:
        counter.update(s.lower().split())

    vocab = {
        "<pad>": 0,
        "<sos>": 1,
        "<eos>": 2,
        "<unk>": 3
    }

    idx = 4
    for word in counter:
        vocab[word] = idx
        idx += 1

    return vocab

def encode(sentence, vocab, add_sos=False, add_eos=False):
    tokens = sentence.lower().split()
    ids = []

    if add_sos:
        ids.append(vocab["<sos>"])

    for t in tokens:
        ids.append(vocab.get(t, vocab["<unk>"]))

    if add_eos:
        ids.append(vocab["<eos>"])

    return torch.tensor([ids])

def decode(ids, vocab):
    inv = {v: k for k, v in vocab.items()}
    return " ".join(inv[i] for i in ids)

# ---------------------------------------------------------
# 2. TOY DATASET
# ---------------------------------------------------------

train_data = [
    ("cats are small animals that live with humans they are independent and like to hunt",
     "cats are independent pets"),

    ("dogs are loyal animals that often live with humans they need exercise and food",
     "dogs are loyal pets"),

    ("the sun is a star it gives heat and light to earth",
     "the sun gives heat and light")
]

input_texts = [x for x, y in train_data]
target_texts = [y for x, y in train_data]

vocab = build_vocab(input_texts + target_texts)

# ---------------------------------------------------------
# 3. ENCODER
# ---------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

# ---------------------------------------------------------
# 4. ATTENTION
# ---------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

# ---------------------------------------------------------
# 5. DECODER WITH ATTENTION
# ---------------------------------------------------------

class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.attention = Attention(hidden_dim)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x)
        attn_weights = self.attention(hidden.permute(1,0,2), encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc(output)
        return prediction, hidden

# ---------------------------------------------------------
# 6. TRAINING SETUP
# ---------------------------------------------------------

encoder = Encoder(len(vocab), 64, 128)
decoder = AttnDecoder(len(vocab), 64, 128)

criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
enc_opt = optim.Adam(encoder.parameters(), lr=0.001)
dec_opt = optim.Adam(decoder.parameters(), lr=0.001)

def train_step(input_tensor, target_tensor):
    encoder_outputs, hidden = encoder(input_tensor)
    loss = 0

    dec_input = torch.tensor([[vocab["<sos>"]]])

    for t in range(1, target_tensor.size(1)):
        output, hidden = decoder(dec_input, hidden, encoder_outputs)
        output = output.squeeze(1)
        loss += criterion(output, target_tensor[:, t])
        dec_input = target_tensor[:, t].unsqueeze(0)

    enc_opt.zero_grad()
    dec_opt.zero_grad()
    loss.backward()
    enc_opt.step()
    dec_opt.step()

    return loss.item()

# ---------------------------------------------------------
# 7. TRAINING LOOP
# ---------------------------------------------------------

print("Training...")
for epoch in range(200):
    total_loss = 0
    for inp, tgt in train_data:
        inp_ids = encode(inp, vocab)
        tgt_ids = encode(tgt, vocab, add_sos=True, add_eos=True)
        loss = train_step(inp_ids, tgt_ids)
        total_loss += loss
    if epoch % 20 == 0:
        print(f"Epoch {epoch} Loss: {total_loss:.4f}")

# ---------------------------------------------------------
# 8. SUMMARIZATION
# ---------------------------------------------------------

def summarize(sentence):
    inp = encode(sentence, vocab)
    encoder_outputs, hidden = encoder(inp)

    dec_input = torch.tensor([[vocab["<sos>"]]])
    summary = []

    for _ in range(20):
        output, hidden = decoder(dec_input, hidden, encoder_outputs)
        token = output.argmax(2).item()

        if token == vocab["<eos>"]:
            break

        summary.append(token)
        dec_input = torch.tensor([[token]])

    return decode(summary, vocab)

# ---------------------------------------------------------
# 9. TEST
# ---------------------------------------------------------
test_sentence = input("How can I help you");
print("Summary:")
print(summarize(test_sentence))

