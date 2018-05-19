from keras.callbacks import Callback
from keras.utils import to_categorical
import numpy as np

def get_batch(dataset, classes):
    epoch = dataset.epoch
    epochs = np.arange(epoch)

    while True:
        np.random.shuffle(epochs)
        for i in epochs:
            inpt_data = dataset.docs[i]
            outpt_data = to_categorical(dataset.label[i],num_classes=classes)
            yield (inpt_data, outpt_data)

class TestHistory(Callback):
    def __init__(self,dataset):
        self.best_acc = []
        self.dataset = dataset
    def on_epoch_end(self, epoch, logs={}):
        score, acc = self.model.evaluate_generator(get_batch(self.dataset),steps=self.dataset.epoch)
        self.best_acc.append(acc)
        print("best test -los:{}  -acc:{}".format(score,acc))
    def on_train_end(self,logs={}):
        print("test acc list: "+str(self.best_acc))
        print("BestTest  acc:{}".format(max(self.best_acc)))

class ValTestLog(Callback):
    def __init__(self,dataset,classes):
        self.val_acc = []
        self.test_acc = []
        self.dataset = dataset
        self.classes = classes
    def on_epoch_end(self, epoch, logs={}):
        acc_val = logs.get('val_acc')
        self.val_acc.append(acc_val)
        score, acc_test = self.model.evaluate_generator(get_batch(self.dataset, self.classes),steps=self.dataset.epoch)
        self.test_acc.append(acc_test)
        print("test -los:{}  -acc:{}".format(score, acc_test))

    def on_train_end(self, logs={}):
        val_test_acc = [(val, test) for val, test in zip(self.val_acc,self.test_acc)]
        val_test_acc = sorted(val_test_acc,key=lambda a:a[0],reverse=True)
        print("BestTestAcc:{}".format(val_test_acc[0]))
        with open("./result", 'a') as f:
            f.write("Model bset val_acc and test_acc:\n")
            f.write(str(val_test_acc[:3])+"\n")
