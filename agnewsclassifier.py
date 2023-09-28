


import numpy as np
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,f1_score,accuracy_score
import matplotlib.pyplot as plt


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
    sentences.append((cstr,label))


data=AG_NEWS(split='test')
for label,stri in data:
    cstr=clean(stri)
    sentences.append((cstr,label))

print(len(sentences))
#print(sentences[150:200])

import random

# maybe remove punctuation and apostrophes in sentences

unknownTokens=[] # tokens which arent in the vocabulary except \n
vocab=[]

# splitting sentences into tokens and gets token count
vocab_count={}
utokens=[] #
for sent in sentences:
    tokens=tokenizer(sent[0])
    sent_tokens=[]
    for tok in tokens:
        tok=str(tok)
        sent_tokens.append(tok)
        vocab_count[tok]=vocab_count.get(tok,0)+1
    utokens.append((sent_tokens,sent[1]))


# Removing infrequent tokens and replacing unknown tokens
max_len=-1
ctokens=[]
for sent in utokens:
    sent_tokens=[]
    for tok in sent[0]:
        if ('\n' not in tok):
            if vocab_count[tok]>2:
                if embeddings.get(tok,-1)==-1:
                    sent_tokens.append('<UNK>')
                    unknownTokens.append(tok)
                else:
                    sent_tokens.append(tok)
            else:
                unknownTokens.append(tok)
    if len(sent_tokens)>max_len:
        max_len=len(sent_tokens)
    ctokens.append((sent_tokens,sent[1]))


print(len(unknownTokens))
unknownTokens=set(unknownTokens)
print('Number of unknown words: ',len(unknownTokens))
# print(max_len)

# Limiting length of sentences
seqLen=75
LTokens=[]
for sent in ctokens:
    if len(sent[0])<=seqLen:
        LTokens.append(sent)

print(len(LTokens))

# returns latest available index for the tokens after parsing
# initially give ix as -1
# set test_set equal to 1 when parsing test set
def Index(ix,tokens):
    # assigning indices to tokens
    for sent in tokens:
        for tok in sent[0]:
            if embeddings.get(tok,-1)!=-1:
                if embeddings[tok][1]==-1:
                    vocab.append(tok)
                    ix+=1
                    embeddings[tok][1]=ix
    return ix



vocab_len=Index(-1,LTokens)
vocab_len+=1
print('Words in vocabulary: ',vocab_len)

# splitting into train,test,val
# Remainder-7600-7600
testData=[]
trainData=[]
valData=[]

random.seed(7)
valData.extend(LTokens[:7600])
testData.extend(LTokens[-7600:])
trainData.extend(LTokens[7600:-7600])
random.shuffle(valData)
random.shuffle(testData)
random.shuffle(trainData)

print("Samples in Train Data: ",len(trainData))
print("Samples in Validation Data: ",len(valData))
print("Samples in Test Data: ",len(testData))
print(len(LTokens))

import torch
from torch import nn

class ClassifierLSTM(nn.Module):
    def __init__(self,embed_size,vocab_len):
        super().__init__()
        # self.weighting=nn.Linear(1200,600)
        self.lstm=nn.LSTM(600,100,batch_first=True) # maybe increase hidden state size
        self.w1=nn.parameter.Parameter(torch.randn(1,dtype=torch.float32))
        self.w2=nn.parameter.Parameter(torch.randn(1,dtype=torch.float32))
        self.classifier=nn.Linear(100,4)

    def forward(self,layer1,layer2,hidden):
        x=torch.add(torch.mul(self.w1,layer1),torch.mul(self.w2,layer2))
        # x=torch.cat((layer1,layer2),dim=2)
        # x=self.weighting(x)
        out,final_hidd=self.lstm(x,hidden)
        # cell state or hidden state
        logits=self.classifier(final_hidd[1][0,:,:])
        return logits


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
        target=torch.zeros(4,dtype=torch.float32)
        start=0
        end=len(self.dataSeq[idx][0])
        for i in range(start,end):
            if i>=start and i<=(end-2):
                tword=torch.from_numpy(self.embed[self.dataSeq[idx][0][i]][0])
                tword=tword.to(torch.float32)
                tword=torch.reshape(tword,(1,300))
                sample=torch.cat((sample,tword),dim=0)

        # padding
        refT=torch.zeros(self.seqLen,300)
        sample=pad_sequence([refT,sample],batch_first=True)[1,:,:]

        target[self.dataSeq[idx][1]-1]=1

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
        target=torch.zeros(4,dtype=torch.float32)
        start=1
        end=len(self.dataSeq[idx][0])
        for i in range(start,end):
            tword=torch.from_numpy(self.embed[self.dataSeq[idx][0][i]][0])
            tword=tword.to(torch.float32)
            tword=torch.reshape(tword,(1,300))
            sample=torch.cat((sample,tword),dim=0)


        # padding
        refT=torch.zeros(self.seqLen,300)
        sample=pad_sequence([refT,sample],batch_first=True)[1,:,:]
        sample=torch.flip(sample,[0])

        target[self.dataSeq[idx][1]-1]=1


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

Cmodel=ClassifierLSTM(300,vocab_len).to(device)
model=torch.load('/home2/raghavd0/elmo/BiLSTM.pth').to(device)
model.eval()

# optimizer and loss function
# this also applies the softmax function to logits
loss_fn = nn.CrossEntropyLoss()

# Adam vs SGD
# optimizer = torch.optim.SGD(Cmodel.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(Cmodel.parameters(), lr=1e-3)

# train and val datasets for backward and forward lstms
batchSize=64
trainF_dl=DataLoader(FLSTMDataSet(trainData,embeddings,vocab_len,seqLen),batch_size=batchSize)
trainB_dl=DataLoader(BLSTMDataSet(trainData,embeddings,vocab_len,seqLen),batch_size=batchSize)
valF_dl=DataLoader(FLSTMDataSet(valData,embeddings,vocab_len,seqLen),batch_size=batchSize)
valB_dl=DataLoader(BLSTMDataSet(valData,embeddings,vocab_len,seqLen),batch_size=batchSize)

for sample,target in trainF_dl:
    print(sample.shape)
    print(target.shape)
    target=torch.flatten(target,start_dim=0,end_dim=1)
    print(target.shape)
    break

# parallelized training

def train(Cmodel,model,FDL,BDL,loss_fn,optimizer):
    Cmodel.train()
    total_loss=0
    count=0
    actual=np.array([])
    predictions=np.array([])
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

        # feeding inputs into Bi-LSTM to get hidden states
        forwardLogits,backwardLogits,forwardLayers,backwardLayers,newHiddenF,newHiddenB=model(sampleF,sampleB,hiddenF,hiddenB)


        # feed into the classification model
        forwardLayers=(forwardLayers[0].detach(),forwardLayers[1].detach())
        backwardLayers=(backwardLayers[0].detach(),backwardLayers[1].detach())
        forwardLayers=(forwardLayers[0][:,:-1,:],forwardLayers[1][:,:-1,:]) # slicing to only include those forward  layer states that have an equivalent backward state
        backwardLayers=(torch.flip(backwardLayers[0][:,:-1,:],[1]),torch.flip(backwardLayers[1][:,:-1,:],[1])) # flipping to match forward states
        layer1Inputs=torch.cat((forwardLayers[0],backwardLayers[0]),dim=2)
        layer2Inputs=torch.cat((forwardLayers[1],backwardLayers[1]),dim=2)

        hiddenC=(torch.randn((1,sampleF.shape[0],100),dtype=torch.float32).to(device),torch.randn((1,sampleF.shape[0],100),dtype=torch.float32).to(device))
        logits=Cmodel(layer1Inputs,layer2Inputs,hiddenC)
        loss=loss_fn(logits,targetF)


        total_loss+=loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        count+=1

        with torch.no_grad():
            lp=nn.Softmax(dim=1)
            predictions=np.append(predictions,torch.argmax(lp(logits),dim=1).numpy(force=True))
            actual=np.append(actual,torch.argmax(targetF,dim=1).numpy(force=True))

        print(f'Train-----Sample{count}\r',end="")


    
    avgloss=total_loss/count
    micro_f1=f1_score(actual,predictions,average='micro')
    macro_f1=f1_score(actual,predictions,average='macro')
    accuracy=accuracy_score(actual,predictions)
    print(f'Average Loss {avgloss} Accuracy {accuracy} Micro_F1 {micro_f1} Macro_F1{macro_f1} \n')
    return (micro_f1,macro_f1,accuracy,avgloss)

# parallelize testing
def test(Cmodel,model,FDL,BDL,loss_fn,optimizer,conf_matrix=False):
    Cmodel.eval()
    total_loss=0
    count=0
    actual=np.array([])
    predictions=np.array([])
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

            # feeding inputs into Bi-LSTM to get hidden states
            forwardLogits,backwardLogits,forwardLayers,backwardLayers,newHiddenF,newHiddenB=model(sampleF,sampleB,hiddenF,hiddenB)

            # feed into the classification model
            forwardLayers=(forwardLayers[0].detach(),forwardLayers[1].detach())
            backwardLayers=(backwardLayers[0].detach(),backwardLayers[1].detach())
            forwardLayers=(forwardLayers[0][:,:-1,:],forwardLayers[1][:,:-1,:]) # slicing to only include those forward  layer states that have an equivalent backward state
            backwardLayers=(torch.flip(backwardLayers[0][:,:-1,:],[1]),torch.flip(backwardLayers[1][:,:-1,:],[1])) # flipping to match forward states
            layer1Inputs=torch.cat((forwardLayers[0],backwardLayers[0]),dim=2)
            layer2Inputs=torch.cat((forwardLayers[1],backwardLayers[1]),dim=2)

            hiddenC=(torch.randn((1,sampleF.shape[0],100),dtype=torch.float32).to(device),torch.randn((1,sampleF.shape[0],100),dtype=torch.float32).to(device))
            logits=Cmodel(layer1Inputs,layer2Inputs,hiddenC)
            loss=loss_fn(logits,targetF)


            total_loss+=loss.item()

            count+=1

            lp=nn.Softmax(dim=1)
            predictions=np.append(predictions,torch.argmax(lp(logits),dim=1).numpy(force=True))
            actual=np.append(actual,torch.argmax(targetF,dim=1).numpy(force=True))

            print(f'Test-----Sample{count}\r',end="")




    avgloss=total_loss/count
    micro_f1=f1_score(actual,predictions,average='micro')
    macro_f1=f1_score(actual,predictions,average='macro')
    accuracy=accuracy_score(actual,predictions)
    print(f'Average Loss {avgloss} Accuracy {accuracy} Micro_F1 {micro_f1} Macro_F1{macro_f1} \n')
    if conf_matrix:
        stats=(micro_f1,macro_f1,accuracy,avgloss,confusion_matrix(actual,predictions))
    else:
        stats=(micro_f1,macro_f1,accuracy,avgloss)
    return stats

import csv


headers=['EpochNumber','TrainMicroF1','TrainMacroF1','TrainAccuracy','TrainAverageLoss','ValMicroF1','ValMacroF1','ValAccuracy','ValAverageLoss']



epochs = 5

# w1=None
# w2=None
# for name,param in Cmodel.named_parameters():
#     if name=='w1':
#         w1=param
#     if name=='w2':
#         w2=param
# print('Weights before:',w1,w2)
# print('Weights before:',Cmodel.w1,Cmodel.w2)
with open('StatsClassifier_test.csv','w') as csvh:
    csvwriter = csv.writer(csvh, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(headers)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_stats=train(Cmodel,model,trainF_dl,trainB_dl,loss_fn,optimizer)
        val_stats=test(Cmodel,model,valF_dl,valB_dl,loss_fn,optimizer)

        row=[t+1]
        row.extend(list(train_stats))
        row.extend(list(val_stats))

        csvwriter.writerow(row)
    print("Done!")

# for name,param in Cmodel.named_parameters():
#     if name=='w1':
#         w1=param
#     if name=='w2':
#         w2=param
# print('Weights after:',w1,w2)
# print('Weights after:',Cmodel.w1,Cmodel.w2)
# model=torch.load('LSTM.pth').to(device)

# test dataloader
testF_dl=DataLoader(FLSTMDataSet(testData,embeddings,vocab_len,seqLen),batch_size=batchSize)
testB_dl=DataLoader(BLSTMDataSet(testData,embeddings,vocab_len,seqLen),batch_size=batchSize)

# testing ---------
test_stats=test(Cmodel,model,testF_dl,testB_dl,loss_fn,optimizer,True)
disp=ConfusionMatrixDisplay(test_stats[-1],display_labels=['World','Sports','Business','Sci/Tech'])
disp.plot()
plt.savefig('confmatrix_labels')


torch.save(Cmodel,'ClassifierLSTM_test.pth')

# todo
# add dropout layers
# add automatic stopping if perplexity is increasing
# save model periodically after every 2 epochs

"""# Testing"""

# from torch.nn.utils.rnn import pad_sequence
# import torch

# ref=torch.zeros(50,300)
# sample=torch.zeros(22,300)
# c=pad_sequence([ref,sample],batch_first=True)
# print(c.shape)
# c=c[1,:,:]
# print(c.shape)
# c[49,299]=1
# print(c)
# c=torch.flip(c[:-1,:],[0])
# print(c.shape)

# d=torch.ones(64,50,300)
# f=torch.zeros(64,50,300)
# s=torch.cat((d,f),dim=2)
# print(s)
# print(s.shape)

# import torch
# b=torch.ones(100,200)
# c=torch.tensor(4,requires_grad=True,dtype=torch.float32)
# print(torch.mul(b,c))

# f=(torch.ones(4)==torch.zeros(4))
# print(f)
# print(f.type(torch.float))

