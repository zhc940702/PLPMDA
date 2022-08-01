# -*- coding: utf-8 -*-
# @Author: chenglinyu
# @Date  : 2019/3/2
# @Desc  : CD-LNLP method
import numpy as np
import sim
from SNF import *

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
            Yd[i,:] = np.sum( np.multiply(wi.reshape((K,1)), Y[dnn_i,:]), axis=0) / Zd
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


def fast_calculate_new(feature_matrix, neighbor_num):
    """
    :param feature_matrix:
    :param neighbor_num: neighbor_num: must be less or equal than n-1 !!!!(n is the row count of feature matrix
    :return:
    """
    iteration_max = 10
    mu = 6
    X = feature_matrix
    alpha = np.power(X, 2).sum(axis=1)
    temp = alpha + alpha.T - 2 * X * X.T
    temp[np.where(temp < 0)] = 0
    distance_matrix = np.sqrt(temp)
    row_num = X.shape[0]
    e = np.ones((row_num, 1))
    distance_matrix = np.array(distance_matrix + np.diag(np.diag(e * e.T * np.inf)))
    sort_index = np.argsort(distance_matrix, kind='mergesort')
    nearest_neighbor_index = sort_index[:, :neighbor_num].flatten()
    nearest_neighbor_matrix = np.zeros((row_num, row_num))
    nearest_neighbor_matrix[np.arange(row_num).repeat(neighbor_num), nearest_neighbor_index] = 1
    C = nearest_neighbor_matrix
    np.random.seed(0)
    W = np.mat(np.random.rand(row_num, row_num), dtype=float)
    W = np.multiply(C, W)
    lamda = mu * e
    P = X * X.T + lamda * e.T
    for q in range(iteration_max):
        Q = W * P
        W = np.multiply(W, P) / Q
        W = np.nan_to_num(W)
    return W


def calculate_linear_neighbor_simi(feature_matrix, neighbor_rate):
    """
    :param feature_matrix:
    :param neighbor_rate:
    :return:
    """
    neighbor_num = int(neighbor_rate * feature_matrix.shape[0])
    return fast_calculate_new(feature_matrix, neighbor_num)


def normalize_by_divide_rowsum(simi_matrix):
    simi_matrix_copy = simi_matrix.copy()
    for i in range(simi_matrix_copy.shape[0]):
        simi_matrix_copy[i, i] = 0
    row_sum_matrix = np.sum(simi_matrix_copy, axis=1)
    result = np.divide(simi_matrix_copy, row_sum_matrix)
    result[np.where(row_sum_matrix == 0)[0], :] = 0
    return result


def complete_linear_neighbor_simi_matrix(train_association_matrix, neighbor_rate):
    b = np.matrix(train_association_matrix)
    final_simi = calculate_linear_neighbor_simi(b, neighbor_rate)
    normalized_final_simi = normalize_by_divide_rowsum(final_simi)
    return normalized_final_simi


def linear_neighbor_predict(train_matrix, alpha):
    iteration_max = 1
    drug_number = train_matrix.shape[0]
    microbe_number = train_matrix.shape[1]
    w_drug = complete_linear_neighbor_simi_matrix(train_matrix, 0.2)
    w_microbe = complete_linear_neighbor_simi_matrix(train_matrix.T, 0.2)
    XX = []
    MS3 = sim.sequencesim_microbe()
    XX.append(MS3)
    XX.append(w_microbe)
    MS = SNF(XX, 3, 5)
    YY = []
    DS2 = sim.atcsim_drug()
    YY.append(DS2)
    YY.append(w_drug)
    DS = SNF(YY, 3, 5)
    w_drug = DS
    w_microbe = MS
    w_drug_eye = np.eye(drug_number)
    w_microbe_eye = np.eye(microbe_number)
    temp0 = w_drug_eye - alpha * w_drug

    for q in range(iteration_max):
        try:
            temp1 = np.linalg.inv(temp0)
        except Exception:
            temp1 = np.linalg.pinv(temp0)
        temp2 = np.dot(temp1, train_matrix)
    prediction_drug = (1 - alpha) * temp2
    temp3 = w_microbe_eye - alpha * w_microbe

    for p in range(iteration_max):
        try:
            temp4 = np.linalg.inv(temp3)
        except Exception:
            temp4 = np.linalg.pinv(temp3)
        temp5 = np.dot(temp4, train_matrix.T)
    temp6 = (1 - alpha) * temp5
    prediction_microbe = temp6.T
    prediction_result = 0.5 * prediction_drug + 0.5 * prediction_microbe
    return prediction_result