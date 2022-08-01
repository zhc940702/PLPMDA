import numpy as np
import copy
def FindDominantSet(W, K):
    m, n = W.shape
    DS = np.zeros((m, n))
    for i in range(m):
        index = np.argsort(W[i, :])[-K:]  # get the closest K neighbors
        DS[i, index] = W[i, index]  # keep only the nearest neighbors

    # normalize by sum
    B = np.sum(DS, axis=1)
    B = B.reshape(len(B), 1)
    DS = DS / B
    return DS


def normalized(W, ALPHA):
    m, n = W.shape
    W = W + ALPHA * np.identity(m)
    return (W + np.transpose(W)) / 2


def SNF(Wall, K, t, ALPHA=1):
    C = len(Wall)
    m, n = Wall[0].shape

    for i in range(C):
        B = np.sum(Wall[i], axis=1)
        len_b = len(B)
        B = B.reshape(len_b, 1)
        Wall[i] = Wall[i] / B
        Wall[i] = (Wall[i] + np.transpose(Wall[i])) / 2

    newW = []

    for i in range(C):
        newW.append(FindDominantSet(Wall[i], K))

    Wsum = np.zeros((m, n))
    for i in range(C):
        Wsum += Wall[i]

    for iteration in range(t):
        Wall0 = []
        for i in range(C):
            temp = np.dot(np.dot(newW[i], (Wsum - Wall[i])), np.transpose(newW[i])) / (C - 1)
            Wall0.append(temp)

        for i in range(C):
            Wall[i] = normalized(Wall0[i], ALPHA)

        Wsum = np.zeros((m, n))
        for i in range(C):
            Wsum += Wall[i]

    W = Wsum / C
    B = np.sum(W, axis=1)
    B = B.reshape(len(B), 1)
    W /= B
    W = (W + np.transpose(W) + np.identity(m)) / 2
    return W


def WKNKN(Y, SD, ST, K, eta):
    Yd = np.zeros(Y.shape)
    Yt = np.zeros(Y.shape)
    wi = np.zeros((K,))
    wj = np.zeros((K,))
    num_drugs, num_targets = Y.shape
    for i in np.arange(num_drugs):
        dnn_i = np.argsort(SD[i,:])[::-1][1:K+1]
        Zd = np.sum(SD[i, dnn_i])
        for ii in np.arange(K):
            wi[ii] = (eta ** (ii)) * SD[i,dnn_i[ii]]
        if not np.isclose(Zd, 0.):
            Yd[i,:] = np.sum(np.multiply(wi.reshape((K,1)), Y[dnn_i,:]), axis=0) / Zd
    for j in np.arange(num_targets):
        tnn_j = np.argsort(ST[j, :])[::-1][1:K+1]
        Zt = np.sum(ST[j, tnn_j])
        for jj in np.arange(K):
            wj[jj] = (eta ** (jj)) * ST[j,tnn_j[jj]]
        if not np.isclose(Zt, 0.):
            Yt[:,j] = np.sum(np.multiply(wj.reshape((1,K)), Y[:,tnn_j]), axis=1) / Zt
    Ydt = (Yd + Yt)/2
    x, y = np.where(Ydt > Y)

    Y_tem = Y.copy()
    Y_tem[x, y] = Ydt[x, y]
    return Y_tem


