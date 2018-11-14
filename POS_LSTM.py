import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
alphbet_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
        for ch in word:
            if ch not in alphbet_to_ix:
                alphbet_to_ix[ch] = len(alphbet_to_ix)

print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# 实际中通常使用更大的维度如32维, 64维.
# 这里我们使用小的维度, 为了方便查看训练过程中权重的变化.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class POS_LSTM(nn.Module):
    '''
    The structure actually only use the word level to produce the POS
    quite simple network
    word -> embedding_word -> lstm() -> output -> softmax
    '''
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size ):
        super(POS_LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, target_size)
        self.hidden=self.init_hidden()

    def init_hidden(self):
        #第一个Varibale是用以初始化hidden_state
        #第二个Varibale, cell_state
        #(num_layers * num_directions, batch, hidden_size)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        return tag_space


class POS_LSTM_CHAR(nn.Module):
    '''
    In this version,
    we add another LSTM layer in order to get the feature on the char level
    The structure of the network will like:
    LSTM(word) -> last_hidden_state, contains the feature of the word
    LSTM2(word + word_char_level_feature) -> output
    output -> softmax
    '''
    def __init__(self, w_embedding_dim, hidden_dim, vocab_size, target_size, c_embedding_dim, alphabet_size):
        super(POS_LSTM_CHAR,self).__init__()

        self.hidden_dim = hidden_dim
        self.c_hidden_dim = c_embedding_dim

        self.char_embedding = nn.Embedding(alphabet_size, c_embedding_dim)
        self.char_lstm = nn.LSTM(c_embedding_dim, c_embedding_dim)

        self.word_embedding = nn.Embedding(vocab_size, w_embedding_dim)
        self.word_lstm = nn.LSTM(w_embedding_dim+c_embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, target_size)
        self.hidden = self.init_hidden(hidden_dim)
        self.hidden_char = self.init_hidden(c_embedding_dim)

    def init_hidden(self,dim):
        #第一个Varibale是用以初始化hidden_state
        #第二个Varibale, cell_state
        #(num_layers * num_directions, batch, hidden_size)
        return (autograd.Variable(torch.zeros(1, 1, dim)),
                autograd.Variable(torch.zeros(1, 1, dim)))

    def forward(self, sentence):
        char_lstm_result=[]
        for idx, word in enumerate(sentence):
            print('char lstm for word {}'.format(index))
            try:
                word_ids = prepare_sequence(word, alphbet_to_ix)
            except:
                print(word)
            char_embeds = self.char_embedding(word_ids)
            self.hidden_char = self.init_hidden(self.c_hidden_dim)
            char_lstm_out, self.hidden_char=self.char_lstm(char_embeds.view(len(word), 1, -1), self.hidden_char)
            char_lstm_result.append(char_lstm_out[-1])

        char_lstm_result = torch.stack(char_lstm_result)
        sentence_in = prepare_sequence(sentence, word_to_ix)
        word_embeds = self.word_embedding(sentence_in)
        word_embeds = word_embeds.view(len(sentence), 1, -1)

        #print('word_embedding size: {} '.format(word_embeds.size()))
        #print('char lstm output size: {}'.format(char_lstm_result.size()))
        inputs = torch.cat((word_embeds, char_lstm_result), 2)

        lstm_out, self.hidden = self.word_lstm(
            inputs, self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return tag_space


#model = POS_LSTM(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(word_to_ix), target_size=len(tag_to_ix))
model = POS_LSTM_CHAR(c_embedding_dim=100, w_embedding_dim=128, hidden_dim=100, vocab_size=len(word_to_ix), target_size=len(tag_to_ix), alphabet_size=len(alphbet_to_ix))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(20):
    print('{} epoch/ {}'.format(epoch+1, 300))
    for index, (sentence, tags) in enumerate(training_data):
        print('sentence index {}'.format(index+1))
        optimizer.zero_grad()

        #because in the next sentence we could not use the same hidden state.
        model.hidden = model.init_hidden(100)
        #sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        tag_score = model(sentence)

        loss = criterion(tag_score, targets)
        loss.backward()
        optimizer.step()

#inputs = prepare_sequence(training_data[0][0],word_to_ix)
tag_scores = model(training_data[0][0])
_, index = torch.max(tag_scores, 1)
print(index)
print(tag_scores)