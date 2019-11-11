import numpy as np
import matplotlib.pyplot as plt
import math
import pickle as pkl
from init import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import multivariate_normal
import time

def lda_init(x_train,y_train,x_test,n):
	lda = LDA(n_components=1)
	x_train=lda.fit_transform(x_train,y_train)
	x_test = lda.transform(x_test)
	
	return x_train,x_test

def lda_nn(x_train,y_train,x_test,y_test,n):
	classifier = KNeighborsClassifier(n_neighbors=1)
	classifier.fit(x_train,y_train)
	print(x_train.shape)
	print(x_test.shape)
	y_pred = classifier.predict(x_test)
	y_pred=y_pred.reshape(-1,1)
	y_temp=y_test.reshape(-1,1)
	y_final=np.concatenate((y_pred,y_temp),axis=1)
	print(y_final)
	return y_final

def lda_gaussian(x_test,y_test,mean_list,cov_list):
	y_pred=[]
	n=y_test.shape[0]
	for i in range(0,n):
		y_pred.append(predict(x_test[i],mean_list,cov_list))
		print(i)
	y_pred=np.array(y_pred)
	print(y_pred)
	print(y_test)
	y_pred=y_pred.flatten().reshape(-1,1)
	y_test=y_test.reshape(-1,1)
	y_final=np.concatenate((y_pred,y_test),axis=1)
	return y_final

def train_dict(x_train,y_train):
	list0=[];list1=[];list2=[];list3=[];list4=[];list5=[];list6=[];list7=[];list8=[];list9=[]
	for idx,label in enumerate(y_train):
		if label==0:list0.append(x_train[idx])
		if label==1:list1.append(x_train[idx])
		if label==2:list2.append(x_train[idx])
		if label==3:list3.append(x_train[idx])
		if label==4:list4.append(x_train[idx])
		if label==5:list5.append(x_train[idx])
		if label==6:list6.append(x_train[idx])
		if label==7:list7.append(x_train[idx])
		if label==8:list8.append(x_train[idx])
		if label==9:list9.append(x_train[idx])
	list0=np.array(list0);mlist0=np.mean(list0,axis=0);clist0=np.cov(list0.T)
	list1=np.array(list1);mlist1=np.mean(list1,axis=0);clist1=np.cov(list1.T)
	list2=np.array(list2);mlist2=np.mean(list2,axis=0);clist2=np.cov(list2.T)
	list3=np.array(list3);mlist3=np.mean(list3,axis=0);clist3=np.cov(list3.T)
	list4=np.array(list4);mlist4=np.mean(list4,axis=0);clist4=np.cov(list4.T)
	list5=np.array(list5);mlist5=np.mean(list5,axis=0);clist5=np.cov(list5.T)
	list6=np.array(list6);mlist6=np.mean(list6,axis=0);clist6=np.cov(list6.T)
	list7=np.array(list7);mlist7=np.mean(list7,axis=0);clist7=np.cov(list7.T)
	list8=np.array(list8);mlist8=np.mean(list8,axis=0);clist8=np.cov(list8.T)
	list9=np.array(list9);mlist9=np.mean(list9,axis=0);clist9=np.cov(list9.T)
	cov_list=[]
	mean_list=np.vstack((mlist0,mlist1,mlist2,mlist3,mlist4,mlist5,mlist6,mlist7,mlist8,mlist9))
	cov_list.extend((clist0,clist1,clist2,clist3,clist4,clist5,clist6,clist7,clist8,clist9))
	cov_list=np.array(cov_list)
	return mean_list,cov_list

def out_final(y_final):
	#pkl.dump(y_final,open("nn1_lda1_preds.pickle","wb"))
	count=0	
	for j in range(y_final.shape[0]):
		if y_final[j,0]==y_final[j,1]:
			count+=1
	print(f'accuaracy is : {count/y_final.shape[0]*100}%')

def main():
	n=10
	tic=time.time()
	x_train,y_train,x_test,y_test=load_data()
	x_train,x_test=lda_init(x_train,y_train,x_test,n)
	mean_list,cov_list=train_dict(x_train,y_train)
	
	#####################
	##UNCOMMENT AS REQD##
	#####################
	
	#y_final_gauss=lda_gaussian(x_test,y_test,mean_list,cov_list)
	#out_final(y_final_gauss)
	
	#y_final_nn=lda_nn(x_train,y_train,x_test,y_test,n)
	#out_final(y_final_nn)
	
	toc=time.time()
	print(f'Time taken is {toc-tic}')

if __name__=="__main__":
	main()