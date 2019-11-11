import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import math
import pickle as pkl
from init import *
import time

def main():
	tic=time.time()
	x_train,y_train,x_test,y_test=load_data()
	#print(x_train.shape)
	#print(y_train.shape)
	#n=y_train.shape[0]
	classifier = KNeighborsClassifier(n_neighbors=9)
	classifier.fit(x_train,y_train)
	#pickle.dump(model, open('nn_model3.sav', 'wb'))
	y_pred = classifier.predict(x_test)
	y_pred=y_pred.reshape(-1,1)
	y_temp=y_test.reshape(-1,1)
	print(y_pred.shape)
	print(y_temp.shape)
	y_final=np.concatenate((y_pred,y_temp),axis=1)
	#pkl.dump(y_final,open("nn9_preds.pickle","wb"))
	count=0	
	for j in range(y_final.shape[0]):
		if y_final[j,0]==y_final[j,1]:
			count+=1
	print(f'accuaracy is : {count/y_final.shape[0]*100}%')
	toc=time.time()
	print(f'Time taken is {toc-tic}')


if __name__=="__main__":
	main()