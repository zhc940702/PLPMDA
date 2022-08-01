import numpy as np
import math
import xlrd
import pandas as pd

path = r"C:\Users\369\Desktop\source_code\dataset"


# microbe similarity
def gausssim_microbe(A, m):
    sum = 0
    for i in range(0, m):
        sum += np.linalg.norm(A[:, i], ord=None, axis=0) ** 2
    gama = 1 / (sum / m)

    MS = np.empty((m, m))
    for i in range(0, m):
        for j in range(0, m):
            MS[i, j] = math.exp(-gama * (np.linalg.norm((A[:, i] - A[:, j]), ord=None) ** 2))
    return MS


def gausssim_microbe2(A, m):
    MS = np.empty((m, m))
    for i in range(0, m):
        for j in range(0, m):
            MS[i, j] = math.exp(-0.1 * (np.linalg.norm((A[:, i] - A[:, j]), ord=None) ** 2))
    return MS


# drug similarity
def gausssim_drug(A, n):
    sum = 0
    for i in range(0, n):
        sum += np.linalg.norm(A[i, :], ord=None) ** 2
    gama = 1 / (sum / n)

    DS = np.empty((n, n))
    for i in range(0, n):
        for j in range(0, n):
            DS[i, j] = math.exp(-gama * (np.linalg.norm((A[i, :] - A[j, :]), ord=None) ** 2))
    return DS


def cosinesim_microbe(A, m):
    MS = np.empty((m, m))
    for i in range(0, m):
        for j in range(0, m):
            MS[i, j] = np.dot(A[:, i], A[:, j]) / (
                        np.linalg.norm(A[:, i], ord=None) * np.linalg.norm(A[:, j], ord=None))
    MS[MS > 1] = 1
    return MS


def cosinesim_drug(A, n):
    DS = np.empty((n, n))
    for i in range(0, n):
        for j in range(0, n):
            DS[i, j] = np.dot(A[i, :], A[j, :]) / (
                        np.linalg.norm(A[i, :], ord=None) * np.linalg.norm(A[j, :], ord=None))
    DS[DS > 1] = 1
    return DS


def atcsim_drug():
    DD = np.loadtxt(path + '/Drug_ATC.txt', dtype=np.float32)
    return DD


def sequencesim_microbe():
    MM = np.loadtxt(path + '/Microbe_sequence.txt', dtype=np.float32)
    return MM



