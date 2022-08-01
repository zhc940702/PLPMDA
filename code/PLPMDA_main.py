from sklearn.model_selection import KFold
import file_rw
import sim
import datetime
import numpy as np
import pandas as pd
import math
import os
import PLPMDA
from SNF import *
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, accuracy_score, precision_recall_curve, \
    average_precision_score
from sklearn.preprocessing import MinMaxScaler



def calculate(alpha):
    print("Performing model---calculate.Please wait a moment.")
    path = r"C:\Users\369\Desktop\source_code\dataset"
    start = datetime.datetime.now()
    A, pp, m, n = file_rw.Amatrix2()
    aucs = []
    auprs = []
    for times in range(50):
        np.random.seed(np.random.randint(1, 100) + times)
        randlist = np.random.permutation(pp)
        test = []
        pred = []
        for cv in range(5):
            if cv != 4:
                partrandlist = randlist[cv * math.floor(pp / 5):(cv + 1) * math.floor(pp / 5)].copy()
            else:
                partrandlist = randlist[cv * math.floor(pp / 5):pp].copy()
            A2, rand_1 = file_rw.A2matrix2_Drugvirus(A, partrandlist)
            XX = []
            YY = []
            MS = sim.gausssim_microbe(A2, m)
            DS = sim.gausssim_drug(A2, n)
            XX.append(MS)
            YY.append(DS)
            MS2 = np.loadtxt(path+ '/Microbe_functional.txt', dtype=np.float32)
            DS2 = np.loadtxt(path+ '/Drug_smile.txt', dtype=np.float32)
            XX.append(MS2)
            YY.append(DS2)
            SM = SNF(XX, 3, 2)
            SD = SNF(YY, 3, 2)
            A2 = WKNKN(A2, SD, SM, 20, 0.9)
            S = PLPMDA.linear_neighbor_predict(A2, alpha=alpha)
            pos = len(partrandlist)
            y_true = np.array([0] * (len(rand_1) - pos) + [1] * pos)
            test = np.append(test, y_true)
            y_pred = np.array([0] * (len(rand_1)))
            y_pred = y_pred.astype('float32')
            for jj in range(0, len(rand_1)):
                y_pred[jj] = y_pred[jj] + (S[int(rand_1[jj][0]), int(rand_1[jj][1])])
            pred = np.append(pred, y_pred)

        aucs.append(roc_auc_score(test, pred))
        auprs.append(average_precision_score(test, pred))

    print("the 5-fold CV of Drugvirus average AUC:")
    print(str(np.mean(aucs)) + "±" + str(np.std(aucs)))
    print(str(np.mean(auprs)) + "±" + str(np.std(auprs)))

    end = datetime.datetime.now()
    print(end-start)



if __name__ == '__main__':
    calculate(0.1)
