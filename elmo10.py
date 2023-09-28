
import numpy as np

# parsing the embeddings
embeddings={} # indicates embedding,one-hot-encoding index
embeddings['<UNK>']=[np.zeros(300,dtype=np.float32),-1] # the unknown token
with open('./glove.6B.300d.txt','r') as ef:
  for line in ef:
    data=line.split()
    embeddings[data[0]]=[np.array(data[1:],dtype=np.float32),-1]



"""## Cleaning the text data"""

import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from spacy.lang.en import English
import re
import portalocker
from torchtext.datasets import AG_NEWS

# using nltk to split text into sentences
# using spacy to tokenize sentences into tokens, nltk was running into some caveats

nlp = English()
tokenizer=nlp.tokenizer
nltk.download('punkt')


# Pre-cleaning the text before splitting into sentences
# This will clean a piece of text
def clean(t):
    # cleaning
    t = re.sub(r'(((http|https):\/\/)|www\.)([a-zA-Z0-9]+\.){0,2}[a-zA-Z0-9]+([a-zA-Z0-9\/#%&=\?_\.\-\+]+)', "", t)
    t = re.sub(r'(@[a-zA-Z0-9_]+)', "", t)
    t = re.sub(r'(#[a-zA-Z0-9_]+\b)', "", t)
    t = re.sub(r'\d+', "",t)
    t = re.sub(r'--'," ",t)
    # special characters
    t = re.sub(r'[\_\$\*\^\(\)\[\]\{\}\=\+\<\>",\&\%\-\—\”\“\–\\\.\?\!;]'," ",t)
    t=re.sub(r'\n'," ",t)
    t=t.lower()
    return t


data=AG_NEWS(split='train')
sentences=[]
for label,stri in data:
    cstr=clean(stri)
    sentences.append(cstr)

# testing purposes
# sentences=sentences[:25000]

# print(len(sentences))
#print(sentences[150:200])

import random

# maybe remove punctuation and apostrophes in sentences

unknownTokens=[] # tokens which arent in the vocabulary except \n
vocab=[]

# splitting sentences into tokens and gets token count
vocab_count={}
utokens=[] #
for sent in sentences:
    tokens=tokenizer(sent)
    sent_tokens=[]
    for tok in tokens:
        tok=str(tok)
        sent_tokens.append(tok)
        vocab_count[tok]=vocab_count.get(tok,0)+1
    utokens.append(sent_tokens)


# Removing infrequent tokens and replacing unknown tokens
ctokens=[]
for sent in utokens:
    sent_tokens=[]
    for tok in sent:
        if ('\n' not in tok):
            if vocab_count[tok]>2:
                if embeddings.get(tok,-1)==-1:
                    sent_tokens.append('<UNK>')
                    unknownTokens.append(tok)
                else:
                    sent_tokens.append(tok)
            else:
                unknownTokens.append(tok)
    ctokens.append(sent_tokens)



unknownTokens=set(unknownTokens)
# print('Number of unknown words: ',len(unknownTokens))

# print(list(dict.fromkeys(unknownTokens))[600:700])

# returns latest available index for the tokens after parsing
# initially give ix as -1
# set test_set equal to 1 when parsing test set
def Index(ix,tokens):
    # assigning indices to tokens
    for sent in tokens:
        for tok in sent:
            if embeddings.get(tok,-1)!=-1:
                if embeddings[tok][1]==-1:
                    vocab.append(tok)
                    ix+=1
                    embeddings[tok][1]=ix
    return ix



vocab_len=Index(-1,ctokens)
vocab_len+=1
# print('Words in vocabulary: ',vocab_len)

# splitting into sequences
seqLen=50
SeqData=[]
for sent in ctokens:
    for i in range(0,len(sent),seqLen):
        if (i+seqLen)<len(sent):
            SeqData.append(sent[i:i+seqLen])
        else:
            SeqData.append(sent[i:])


# splitting into train,test,val
# 70-20-10
testData=[]
trainData=[]
valData=[]
trainval=[]

random.seed(7)
random.shuffle(SeqData)
trainData.extend(SeqData[:int(0.7*len(SeqData))])
testData.extend(SeqData[int(0.7*len(SeqData)):int(0.9*len(SeqData))])
valData.extend(SeqData[int(0.9*len(SeqData)):])


print("Samples in Train Data: ",len(trainData))
print("Samples in Validation Data: ",len(valData))
print("Samples in Test Data: ",len(testData))
print(len(SeqData))

import torch
from torch import nn

class BiLSTM(nn.Module):
    def __init__(self,embed_size,vocab_len):
        super().__init__()
        self.forward1=nn.LSTM(embed_size,embed_size,batch_first=True)
        self.forward2=nn.LSTM(embed_size,embed_size,batch_first=True)
        self.backward1=nn.LSTM(embed_size,embed_size,batch_first=True)
        self.backward2=nn.LSTM(embed_size,embed_size,batch_first=True)
        self.forwardL=nn.Linear(embed_size,vocab_len)
        self.backwardL=nn.Linear(embed_size,vocab_len)

    # returns logits and final hidden state
    # hiddenF= (hiddenLayer1F,hiddenLayer2F)
    # hiddenB= (hiddenLayer1B,hiddenLayer2B)
    # returns forward logits,backward logits,forward layers,backward layers,forward hiddens,backward hiddens
    def forward(self,xF,xB,hiddenF,hiddenB):
        # forward LSTM
        layerF1,hiddenF1=self.forward1(xF,hiddenF[0])
        layerF2,hiddenF2=self.forward2(layerF1,hiddenF[1])
        newHiddenF=(hiddenF1,hiddenF2)
        forwardLayers=(layerF1,layerF2)

        # backward LSTM
        layerB1,hiddenB1=self.backward1(xB,hiddenB[0])
        layerB2,hiddenB2=self.backward2(layerB1,hiddenB[1])
        newHiddenB=(hiddenB1,hiddenB2)
        backwardLayers=(layerB1,layerB2)

        # generating logits
        forwardLogits=self.forwardL(layerF2)
        backwardLogits=self.backwardL(layerB2)


        return forwardLogits,backwardLogits,forwardLayers,backwardLayers,newHiddenF,newHiddenB

from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence

class FLSTMDataSet(Dataset):
    def __init__(self,data,embed,vocab_len,seqLen):
        self.embed=embed
        self.dataSeq=data
        self.vocab_len=vocab_len
        self.seqLen=seqLen

    def __len__(self):
        return len(self.dataSeq)


    # returns sample,target
    def __getitem__(self,idx):

        # getting forward train samples
        sample=torch.empty(0,300,dtype=torch.float32)
        target=torch.empty(0,self.vocab_len,dtype=torch.float32)
        start=0
        end=len(self.dataSeq[idx])
        for i in range(start,end):
            if i>=start and i<=(end-2):
                tword=torch.from_numpy(self.embed[self.dataSeq[idx][i]][0])
                tword=tword.to(torch.float32)
                tword=torch.reshape(tword,(1,300))
                sample=torch.cat((sample,tword),dim=0)
            if i>=(start+1):
                targetWord=torch.zeros(self.vocab_len,dtype=torch.float32)
                targetWord[self.embed[self.dataSeq[idx][i]][1]]=1
                targetWord=torch.reshape(targetWord,(1,self.vocab_len))
                target=torch.cat((target,targetWord),dim=0)

        # padding
        refT=torch.zeros(self.seqLen,300)
        sample=pad_sequence([refT,sample],batch_first=True)[1,:,:]

        refT=torch.zeros(self.seqLen,self.vocab_len)
        target=pad_sequence([refT,target],batch_first=True)[1,:,:]

        return sample,target


class BLSTMDataSet(Dataset):
    def __init__(self,data,embed,vocab_len,seqLen):
        self.embed=embed
        self.dataSeq=data
        self.vocab_len=vocab_len
        self.seqLen=seqLen

    def __len__(self):
        return len(self.dataSeq)


    # returns sample,target
    def __getitem__(self,idx):

        # getting forward train samples
        sample=torch.empty(0,300,dtype=torch.float32)
        target=torch.empty(0,self.vocab_len,dtype=torch.float32)
        start=len(self.dataSeq[idx])-1
        end=-1
        for i in range(start,end,-1):
            if i<=start and i>=(end+2):
                tword=torch.from_numpy(self.embed[self.dataSeq[idx][i]][0])
                tword=tword.to(torch.float32)
                tword=torch.reshape(tword,(1,300))
                sample=torch.cat((sample,tword),dim=0)
            if i<=(start-1):
                targetWord=torch.zeros(self.vocab_len,dtype=torch.float32)
                targetWord[self.embed[self.dataSeq[idx][i]][1]]=1
                targetWord=torch.reshape(targetWord,(1,self.vocab_len))
                target=torch.cat((target,targetWord),dim=0)

        # padding
        refT=torch.zeros(self.seqLen,300)
        sample=pad_sequence([refT,sample],batch_first=True)[1,:,:]

        refT=torch.zeros(self.seqLen,self.vocab_len)
        target=pad_sequence([refT,target],batch_first=True)[1,:,:]

        return sample,target

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model=BiLSTM(300,vocab_len).to(device)

# optimizer and loss function
# this also applies the softmax function to logits
loss_fn = nn.CrossEntropyLoss()

# Adam vs SGD
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# train and val datasets for backward and forward lstms
batchSize=64
trainF_dl=DataLoader(FLSTMDataSet(trainData,embeddings,vocab_len,seqLen),batch_size=batchSize)
trainB_dl=DataLoader(BLSTMDataSet(trainData,embeddings,vocab_len,seqLen),batch_size=batchSize)
valF_dl=DataLoader(FLSTMDataSet(valData,embeddings,vocab_len,seqLen),batch_size=batchSize)
valB_dl=DataLoader(BLSTMDataSet(valData,embeddings,vocab_len,seqLen),batch_size=batchSize)

# for sample,target in trainF_dl:
#     print(sample.shape)
#     print(target.shape)
#     target=torch.flatten(target,start_dim=0,end_dim=1)
#     print(target.shape)
#     break

# parallelized training

def train(model,FDL,BDL,loss_fn,optimizer):
    model.train()
    total_lossF=0
    total_lossB=0
    count=0
    for ((sampleF,targetF),(sampleB,targetB)) in zip(FDL,BDL):

        # hidden states for the Bi-LSTM
        hiddenF1=(torch.randn((1,sampleF.shape[0],300),dtype=torch.float32).to(device),torch.randn((1,sampleF.shape[0],300),dtype=torch.float32).to(device))
        hiddenF2=(torch.randn((1,sampleF.shape[0],300),dtype=torch.float32).to(device),torch.randn((1,sampleF.shape[0],300),dtype=torch.float32).to(device))
        hiddenB1=(torch.randn((1,sampleB.shape[0],300),dtype=torch.float32).to(device),torch.randn((1,sampleB.shape[0],300),dtype=torch.float32).to(device))
        hiddenB2=(torch.randn((1,sampleB.shape[0],300),dtype=torch.float32).to(device),torch.randn((1,sampleB.shape[0],300),dtype=torch.float32).to(device))
        hiddenF=(hiddenF1,hiddenF2)
        hiddenB=(hiddenB1,hiddenB2)

        sampleF,targetF=sampleF.to(device),targetF.to(device)
        sampleB,targetB=sampleB.to(device),targetB.to(device)

        # feeding inputs in
        forwardLogits,backwardLogits,forwardLayers,backwardLayers,newHiddenF,newHiddenB=model(sampleF,sampleB,hiddenF,hiddenB)
        forwardLogits,targetF=torch.flatten(forwardLogits,start_dim=0,end_dim=1),torch.flatten(targetF,start_dim=0,end_dim=1)
        backwardLogits,targetB=torch.flatten(backwardLogits,start_dim=0,end_dim=1),torch.flatten(targetB,start_dim=0,end_dim=1)

        # calculating loss
        lossF=loss_fn(forwardLogits,targetF)
        lossB=loss_fn(backwardLogits,targetB)
        # print(loss.item())
        total_lossF+=lossF.item()
        total_lossB+=lossB.item()

        lossF.backward()
        lossB.backward()
        optimizer.step()
        optimizer.zero_grad()
        count+=1
        with torch.no_grad():
            print(f'Train-----Batch {count} Forward ----> {torch.exp(lossF).item()} Backward ----> {torch.exp(lossB).item()}\n',end="")



    perpF=torch.exp(torch.tensor(total_lossF/len(FDL),dtype=torch.float32)).item()
    perpB=torch.exp(torch.tensor(total_lossB/len(BDL),dtype=torch.float32)).item()
    print(f'Forward Perplexity: {perpF}, Backward Perplexity: {perpB}\n')
    return perpF,perpB

# parallelize testing
def test(model,FDL,BDL,loss_fn,optimizer):
    model.eval()
    total_lossF=0
    total_lossB=0
    count=0
    with torch.no_grad():
        for ((sampleF,targetF),(sampleB,targetB)) in zip(FDL,BDL):

            # hidden states for the Bi-LSTM
            hiddenF1=(torch.randn((1,sampleF.shape[0],300),dtype=torch.float32).to(device),torch.randn((1,sampleF.shape[0],300),dtype=torch.float32).to(device))
            hiddenF2=(torch.randn((1,sampleF.shape[0],300),dtype=torch.float32).to(device),torch.randn((1,sampleF.shape[0],300),dtype=torch.float32).to(device))
            hiddenB1=(torch.randn((1,sampleB.shape[0],300),dtype=torch.float32).to(device),torch.randn((1,sampleB.shape[0],300),dtype=torch.float32).to(device))
            hiddenB2=(torch.randn((1,sampleB.shape[0],300),dtype=torch.float32).to(device),torch.randn((1,sampleB.shape[0],300),dtype=torch.float32).to(device))
            hiddenF=(hiddenF1,hiddenF2)
            hiddenB=(hiddenB1,hiddenB2)

            sampleF,targetF=sampleF.to(device),targetF.to(device)
            sampleB,targetB=sampleB.to(device),targetB.to(device)

            # feeding inputs in
            forwardLogits,backwardLogits,forwardLayers,backwardLayers,newHiddenF,newHiddenB=model(sampleF,sampleB,hiddenF,hiddenB)
            forwardLogits,targetF=torch.flatten(forwardLogits,start_dim=0,end_dim=1),torch.flatten(targetF,start_dim=0,end_dim=1)
            backwardLogits,targetB=torch.flatten(backwardLogits,start_dim=0,end_dim=1),torch.flatten(targetB,start_dim=0,end_dim=1)

            # calculating loss
            lossF=loss_fn(forwardLogits,targetF)
            lossB=loss_fn(backwardLogits,targetB)
            total_lossF+=lossF.item()
            total_lossB+=lossB.item()

            count+=1
            with torch.no_grad():
                print(f'Test-----Batch {count} Forward ----> {torch.exp(lossF).item()} Backward ----> {torch.exp(lossB).item()}\n',end="")


    perpF=torch.exp(torch.tensor(total_lossF/len(FDL),dtype=torch.float32)).item()
    perpB=torch.exp(torch.tensor(total_lossB/len(BDL),dtype=torch.float32)).item()
    print(f'Forward Perplexity: {perpF}, Backward Perplexity: {perpB}\n')
    return perpF,perpB

import csv


headers=['EpochNumber','ForwardTrainAveragePerplexity','BackwardTrainAveragePerplexity','ForwardValAveragePerplexity','BackwardValAveragePerplexity']



epochs = 10
with open('Stats.csv','w') as csvh:
    csvwriter = csv.writer(csvh, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(headers)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_stats=train(model,trainF_dl,trainB_dl,loss_fn,optimizer)
        print("Validating......")
        val_stats=test(model,valF_dl,valB_dl,loss_fn,optimizer)

        row=[t+1]
        row.extend(list(train_stats))
        row.extend(list(val_stats))

        csvwriter.writerow(row)

        # saving model after 2 epochs
        if (t+1)%2==0:
            torch.save(model,'BiLSTM'+str(t+1)+'epoch.pth')
    print("Done!")



# test dataloader
testF_dl=DataLoader(FLSTMDataSet(testData,embeddings,vocab_len,seqLen),batch_size=batchSize)
testB_dl=DataLoader(BLSTMDataSet(testData,embeddings,vocab_len,seqLen),batch_size=batchSize)

# testing ---------
print("Testing-------------------")
test_stats=test(model,testF_dl,testB_dl,loss_fn,optimizer)


torch.save(model,'BiLSTM.pth')

# todo
# add dropout layers


"""# Testing"""

# from torch.nn.utils.rnn import pad_sequence
# import torch

# ref=torch.zeros(50,300)
# sample=torch.zeros(22,300)
# c=pad_sequence([ref,sample],batch_first=True)
# print(c.shape)
# c=c[1,:,:]
# print(c.shape)