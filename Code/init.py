import numpy as np
import sys
sys.path.insert(1, '../fashion_mnist/utils')
import mnist_reader
import matplotlib.pyplot as plt
import math
import pickle as pkl

def load_data():
	x_train,y_train=mnist_reader.load_mnist('../fashion_mnist/data/fashion',kind='train')

	x_test,y_test=mnist_reader.load_mnist('../fashion_mnist/data/fashion',kind='t10k')

	print(f'x train shape: {x_train.shape}')
	print(f'y train shape: {y_train.shape} with labels {set(y_train)}')
	print(f'x test shape: {x_test.shape}')
	print(f'y test shape: {y_test.shape} with labels {set(y_test)}')
	print(f'number of x_train: {x_train.shape[0]} ')
	print(f'number of x_test: {x_test.shape[0]} ')

	return x_train,y_train,x_test,y_test

def visualize_data(x_train,y_train):
	index = np.random.randint(x_train.shape[0],size=9)
	for i in range(0,9):
		plt.subplot(3,3,i+1)
		plt.imshow(x_train[index[i]].reshape(28,28))
		plt.title(f'Index: {index[i]}, Label: {y_train[index[i]]} ')
		plt.colorbar()
		plt.grid(False)
		plt.tight_layout()
	plt.show()

def gaussian(x, mu, sig):
	k=784
	gauss=((2*math.pi)**(-k/2))*(np.linalg.det(sig)**(-1/2))*(np.exp((-1/2)*((x-mu).T)*np.linalg.inv(sig)*(x-mu)))
	return gauss

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

def norm(ls):
	for i in ls:
		max_e=np.amax(ls)
		ls/=max_e
	return ls

def reg(ls,n):
	reg_mat=np.eye(784)
	ls+=n*reg_mat
	return ls

def predict(x1,mu1,cov1):
	from scipy.stats import multivariate_normal
	y_pred=[]
	for i in range(0,mu1.shape[0]):
		# print(x1.shape)
		# print(mu1[i].shape)
		# print(cov1[i].shape)
		y_pred.append(multivariate_normal.logpdf(x1, mean=mu1[i], cov=cov1[i]))

	y_pred=np.array(y_pred)
	#print(y_pred)
	#maxindex = y_pred.argmax()
	maxindex = np.where(y_pred == np.amax(y_pred))
	#print(maxindex[0])

	return maxindex[0]

def void_main():
	x_train,y_train,x_test,y_test=load_data()
	#visualize_data(x_train,y_train)
	#mean_list,cov_list=train_dict(x_train,y_train)
	# pkl.dump(mean_list,open("mean_list.pickle","wb"))
	# pkl.dump(cov_list,open("cov_list.pickle","wb"))
	mean_list = pkl.load(open("../Datadumps/Q1/mean_list.pickle","rb"))
	cov_list = pkl.load(open("../Datadumps/Q1/cov_list.pickle","rb"))
	cov_list=norm(cov_list)
	cov_list=reg(cov_list,0.88)
	y_pred=[]
	count=0
	for i in range(0,cov_list.shape[0]):
		print(np.linalg.det(cov_list[i]))
	
	for i in range(0,y_test.shape[0]):
		y_pred.append(predict(x_test[i],mean_list,cov_list))
		print(i)
	y_pred=np.array(y_pred)
	y_pred=y_pred.flatten().reshape(-1,1)
	y_test=y_test.reshape(-1,1)
	y_final=np.concatenate((y_pred,y_test		),axis=1)
	#pkl.dump(y_final,open("final_preds.pickle","wb"))	

	
def main():
	#void_main()
	x_train,y_train,x_test,y_test=load_data()
	visualize_data(x_train,y_train)
	mean_list = pkl.load(open("../Datadumps/Q1/mean_list.pickle","rb"))
	cov_list = pkl.load(open("../Datadumps/Q1/cov_list.pickle","rb"))
	y_final=pkl.load(open("../Datadumps/Q1/final_preds.pickle","rb"))
	count=0
	for j in range(y_final.shape[0]):
		if y_final[j,0]==y_final[j,1]:
			count+=1
	print(f'accuaracy is : {count/y_final.shape[0]*100}%')


if __name__=="__main__":
	main()	