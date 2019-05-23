# Import packages
# download nltk only once
# import nltk
# nltk.download('punkt')
import numpy
import torch
from models import InferSent
import csv

# Set up infersent
V = 1 # 1: GloVe    2: fastText
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))

# Select word to vector path file
if V == 1:
    W2V_PATH = 'dataset/GloVe/glove.840B.300d.txt' # GloVe
else:
    W2V_PATH = 'dataset/fastText/crawl-300d-2M.vec' # fastText
infersent.set_w2v_path(W2V_PATH)

# Choose number of words to add to vocabulary
infersent.build_vocab_k_words(K=100000)

# Import sentences
sentences = []
labels = []
with open('dataset/TrainingData/articles_with_biases1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for line in csv_reader:
        num = line[0]
        sent = line[1]
        bias = line[2]
        # sentences.append(line.strip())
        print('Number is: ' + str(num))
        print('Sentence: ' + sent)
        print('bias: ' + str(bias))
        break
sentences = sentences[:10]
print('{0} sentences will be encoded to embeddings.'.format(len(sentences)))

# Extract vocabulary from sentences
infersent.build_vocab(sentences, tokenize=True)

# Convert to word embeddings
# embeddings are an ndarray of size: [len(sentences), 4096]
embeddings = infersent.encode(sentences, tokenize=True)
print('nb sentences encoded : {0}.'.format(len(embeddings)))

# Write array to csv
numpy.savetxt("sample_embeddings.csv", embeddings, delimiter=",")