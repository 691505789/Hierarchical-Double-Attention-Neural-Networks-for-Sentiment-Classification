#-*- coding: UTF-8 -*-  
import copy

import numpy
from keras.preprocessing.sequence import pad_sequences


def genBatch(data):
    m = 820
    maxsentencenum = len(data[0])
    for doc in data:
        for sentence in doc:
            if len(sentence) > m:
                m = len(sentence)
        for i in xrange(maxsentencenum - len(doc)):
            doc.append([-1])
    tmp = map(lambda doc: numpy.asarray(map(lambda sentence : sentence + [-1]*(m - len(sentence)), doc), dtype = numpy.int32), data)
    tmp = map(lambda t : t+1, tmp)
    return numpy.asarray(tmp)
            
def genLenBatch(lengths,maxsentencenum):
    lengths = map(lambda length : numpy.asarray(length + [1.0]*(maxsentencenum-len(length)), dtype = numpy.float32)+numpy.float32(1e-4),lengths)
    return reduce(lambda x,y : numpy.concatenate((x,y),axis = 0),lengths)

def genwordmask(docsbatch):
    mask = copy.deepcopy(docsbatch)
    mask = map(lambda x : map(lambda y : [1.0 ,0.0][y == -1],x), mask)
    mask = numpy.asarray(mask,dtype=numpy.float32)
    return mask

def gensentencemask(sentencenum):
    maxnum = sentencenum[0]
    mask = numpy.asarray(map(lambda num : [1.0]*num + [0.0]*(maxnum - num),sentencenum), dtype = numpy.float32)
    return mask.T

class Dataset(object):
    def __init__(self, filename, emb,maxbatch = 32,maxword = 500):
        lines = map(lambda x: x.split('\t\t'), open(filename).readlines())           
        label = numpy.asarray(
            map(lambda x: int(x[2])-1, lines),
            dtype = numpy.int32
        )
        docs = map(lambda x: x[3][0:len(x[3])-1], lines) 
        docs = map(lambda x: x.split('<sssss>'), docs) 
        docs = map(lambda doc: map(lambda sentence: sentence.split(' '),doc),docs)
        docs = map(lambda doc: map(lambda sentence: filter(lambda wordid: wordid !=-1,map(lambda word: emb.getID(word),sentence)),doc),docs)
        tmp = zip(docs, label)
        #random.shuffle(tmp)
        tmp.sort(lambda x, y: len(y[0]) - len(x[0]))  
        docs, label = zip(*tmp)

        # sentencenum = map(lambda x : len(x),docs)
        # length = map(lambda doc : map(lambda sentence : len(sentence), doc), docs)
        self.epoch = len(docs) / maxbatch
        if len(docs) % maxbatch != 0:
            self.epoch += 1
        
        self.docs = []
        self.label = []
        self.length = []

        for i in xrange(self.epoch):
            docsbatch = genBatch(docs[i*maxbatch:(i+1)*maxbatch])
            self.docs.append(docsbatch)
            self.label.append(numpy.asarray(label[i*maxbatch:(i+1)*maxbatch], dtype = numpy.int32))


class Wordlist(object):
    def __init__(self, filename, maxn = 100000):
        lines = map(lambda x: x.split(), open(filename).readlines()[:maxn])
        self.size = len(lines)

        self.voc = [(item[0][0], item[1]) for item in zip(lines, xrange(self.size))]
        self.voc = dict(self.voc)

    def getID(self, word):
        try:
            return self.voc[word]
        except:
            return -1

def padDocs(dataset):
    for indx in range(dataset.epoch):
        docs = []
        for doc in dataset.docs[indx]:
            doc_pad = pad_sequences(doc,maxlen=130, truncating='post')
            docs.append(doc_pad)
        dataset.docs[indx] = numpy.asarray(docs)
    return dataset

# dataname = "IMDB"
# classes = 10
# voc = Wordlist('../data/'+dataname+'/wordlist.txt')
#
# print 'data loadeding....'
# trainset = Dataset('../data/'+dataname+'/test.txt', voc)
# trainset = padDocs(trainset)
# print trainset.docs[3].shape
# print trainset.docs
# f = open('../data/IMDB/testset.save','wb')
# cPickle.dump(trainset, f, protocol=cPickle.HIGHEST_PROTOCOL)
# f.close()
# print 'data load finish...'

'''
lines = map(lambda x: x.split('\t\t'), open('../data/IMDB/test.txt').readlines())
label = numpy.asarray(
    map(lambda x: int(x[2]) - 1, lines),
    dtype=numpy.int32
)
docs = map(lambda x: x[3][0:len(x[3]) - 1], lines)
docs = map(lambda x: x.split('<sssss>'), docs)
docs = map(lambda doc: map(lambda sentence: sentence.split(' '), doc), docs)
length = map(lambda doc: map(lambda sentence: len(sentence), doc), docs)
maxsentencelen = max(map(lambda doc: max(doc), length))

import nltk
fdist = nltk.FreqDist()
fdist_sent = nltk.FreqDist()
totalsentlen = 0
for doc in length:
    doclen = len(doc)
    fdist_sent[doclen] += 1
    # for senlen in doc:
    #     totalsentlen += senlen
    #     fdist[senlen] += 1

print fdist_sent.keys()
print len(fdist_sent.keys())
print sum(fdist_sent.values())
print fdist_sent.plot(74, cumulative=True)
'''
# print len(fdist.keys())
# items = sorted(fdist.items(), lambda a,b: a[1] - b[1])
# print sum(fdist.values())
# print items
# # print fdist.items()
# print maxsentencelen
# print totalsentlen //sum(fdist.values())
# fdist.plot(225, cumulative=True)


