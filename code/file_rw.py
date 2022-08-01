import numpy as np
import math
import xlrd
import xlsxwriter
import xlwt
import random
file_2=xlrd.open_workbook(r"C:\Users\369\Desktop\source_code\dataset\microbe-disease(Drugvirus).xlsx")
file_2.sheet_by_name("Sheet1")

def Amatrix():
	pp=file.sheet_by_name("Sheet1").nrows
	qq=file.sheet_by_name("Sheet1").ncols
	view=np.empty((pp,qq))
	for i in range(0,pp):
		view[i,0]=file.sheet_by_name("Sheet1").row_values(i)[0]
		view[i,1]=file.sheet_by_name("Sheet1").row_values(i)[1]
	n=int(view.max(axis=0)[0])
	m=int(view.max(axis=0)[1])
	A=np.zeros((n,m))
	for i in range(0,pp):
		A[int(view[i,0])-1,int(view[i,1])-1]=1
	return A,pp,m,n

def Amatrix2():
	pp=file_2.sheet_by_name("Sheet1").nrows
	qq=file_2.sheet_by_name("Sheet1").ncols
	view=np.empty((pp,qq))
	for i in range(0,pp):
		view[i,0]=file_2.sheet_by_name("Sheet1").row_values(i)[0]
		view[i,1]=file_2.sheet_by_name("Sheet1").row_values(i)[1]
	n=int(view.max(axis=0)[0])
	m=int(view.max(axis=0)[1])
	A=np.zeros((n,m))
	for i in range(0,pp):
		A[int(view[i,0])-1,int(view[i,1])-1]=1
	return A,pp,m,n


def matrix_save(S,m,n,path,num):
	workbook = xlsxwriter.Workbook(path+str(num)+".xlsx")
	worksheet = workbook.add_worksheet()
	for i in range(n):
		for j in range(m):
			worksheet.write(i,j,S[i,j])
	workbook.close()



def A2matrix2_Drugvirus(A,partrandlist):
	A2=A.copy()
	rand = np.zeros((len(partrandlist)+int(len(partrandlist)), 2))
	cnt = 0
	while (cnt < int(len(partrandlist))):
		a = random.randint(0, A.shape[0] - 1)
		b = random.randint(0, A.shape[1] - 1)
		if A[a, b] != 1 and A2[a, b] != 1:
			rand[cnt, 0] = a
			rand[cnt, 1] = b
			cnt += 1
	for ii in partrandlist:
		A2[int(file_2.sheet_by_name("Sheet1").row_values(ii)[0])-1,int(file_2.sheet_by_name("Sheet1").row_values(ii)[1])-1]=0
		rand[cnt][0] = int(file_2.sheet_by_name("Sheet1").row_values(ii)[0])-1
		rand[cnt][1] = int(file_2.sheet_by_name("Sheet1").row_values(ii)[1])-1
		cnt = cnt+1
	return A2, rand

