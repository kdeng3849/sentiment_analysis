import json
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import spacy

from django.http.response import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from torchtext import data
from torchtext import datasets

from .avg_model import FastText, binary_accuracy, generate_bigrams, predict_sentiment, count_parameters, train, evaluate, epoch_time, model

# Create your views here.
@csrf_exempt
def predict(request):
    # Set to false to disable training, important for when you set up the post request
    # TO_TRAIN = True
    # TO_TRAIN = False

    # # The more epochs the more training will be done. Note that without a GPU, each epoch will take upwards 10 minutes.
    # N_EPOCHS = 5

    # # File name for the model
    FILE_NAME = 'avg_model.pt'

    body = json.loads(request.body.decode('utf-8'))
    sentence = body['sentence']


    # SEED = 1234
    # torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True

    # TEXT = data.Field(tokenize = 'spacy', preprocessing = generate_bigrams)
    # LABEL = data.LabelField(dtype = torch.float)

    # train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    # print(train_data.split(random_state = random.seed(SEED)))
    # train_data, valid_data = train_data.split(random_state = random.seed(SEED))

    # MAX_VOCAB_SIZE = 25_000

    # TEXT.build_vocab(train_data,
    #                 max_size = MAX_VOCAB_SIZE,
    #                 vectors = "glove.6B.100d",
    #                 unk_init = torch.Tensor.normal_)

    # LABEL.build_vocab(train_data)

    # BATCH_SIZE = 64

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    #     (train_data, valid_data, test_data),
    #     batch_size = BATCH_SIZE,
    #     device = device)

    # INPUT_DIM = len(TEXT.vocab)
    # EMBEDDING_DIM = 100
    # OUTPUT_DIM = 1
    # PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    # model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)

    # pretrained_embeddings = TEXT.vocab.vectors

    # model.embedding.weight.data.copy_(pretrained_embeddings)

    # UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    # model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    # model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    # optimizer = optim.Adam(model.parameters())

    # criterion = nn.BCEWithLogitsLoss()

    # model = model.to(device)
    # criterion = criterion.to(device)

    model.load_state_dict(torch.load(FILE_NAME))
    # test_loss, test_acc = evaluate(model, test_iterator, criterion)
    nlp = spacy.load('en')

    # sentiment = predict_sentiment(model, sentence)
    result = {
        "result": predict_sentiment(model, sentence),
    }
    return JsonResponse(result)