import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from tqdm import tqdm
torch.manual_seed(1)

CONTEXT_SIZE = 2  # 左右各2个单词
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# 通过从 `raw_text` 得到一组单词, 进行去重操作
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self, num_voc, num_hidden, context_size, embedding_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(num_voc, embedding_size)
        self.fc1 = nn.Linear(2*context_size*embedding_size, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_voc)

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        inputs = inputs.view(1,-1)
        inputs = F.relu(self.fc1(inputs))
        inputs = F.relu(self.fc2(inputs))
        return inputs

# 创建模型并且训练. 这里有一些函数可以在使用模型之前帮助你准备数据


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


def train_model(model, criterion, optimizer, data, num_epoch=10):
    begin = time.time()
    for i in range(num_epoch):
        print('Epoch {}/{}'.format(i+1, num_epoch))
        print('-'*10)
        running_loss = 0.0
        for (context, target) in tqdm(data):
            input_context = make_context_vector(context, word_to_ix)
            target = make_context_vector([target], word_to_ix)
            #print((input_context))
            optimizer.zero_grad()
            outputs = model(input_context)
            loss = criterion(outputs, target)
            running_loss += loss.data[0]
        epoch_loss = running_loss / len(data)

        print('{} Loss :{:.4f}'.format('Train', epoch_loss))

#make_context_vector(data[0][0], word_to_ix)  # 例子


model = CBOW(vocab_size, num_hidden=100, context_size=2,embedding_size=128)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_model(model, criterion, optimizer, data)
