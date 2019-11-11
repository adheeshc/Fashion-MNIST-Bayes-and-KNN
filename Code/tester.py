import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


def get_accuracy():
	#y_final=pkl.load(open("../Datadumps/Gauss/final_preds.pickle","rb"))
	#y_final=pkl.load(open("../Datadumps/KNN/nn3_preds.pickle","rb"))
	#y_final=pkl.load(open("../Datadumps/LDA/Gauss/gauss_lda9_preds.pickle","rb"))
	#y_final=pkl.load(open("../Datadumps/PCA/Gauss/gauss_pca98_preds.pickle","rb"))
	count=0	
	for j in range(y_final.shape[0]):
		if y_final[j,0]==y_final[j,1]:
			count+=1
	print(f'accuracy is : {count/y_final.shape[0]*100}%')

def plot_gauss_pca():
	x1=np.array([75,80,85,90,95,98]) #REDUCED DIMS
	y1=np.array([77.23,79.06,79.87,80.13,76.58,74.0]) #ACCURACY
	z1=np.array([61.725,151.514,259.594,736.744,2024.279,4722.184]) #TIME

	plt.plot(x1,z1,'-o')
	plt.xlabel("Explained Variance")
	plt.ylabel("Time (secs)")
	plt.title("Explained Variance vs Time")
	plt.show()

	plt.plot(x1,y1,'-o')
	plt.xlabel("Explained Variance")
	plt.ylabel("Accuracy")
	plt.title("Explained Variance vs Accuracy")
	plt.show()

def plot_gauss_lda():
	x1=np.array([1,2,3,4,5,6,7,8,9]) #REDUCED DIMNS
	y1=np.array([47.73,60.13,67.47,69.95,74.16,77.27,77.96,80.44,81.99]) #ACCURACY
	z1=np.array([35.961,34.841,45.367,45.243,42.041,49.465,54.603,52.533,51.532]) #TIME

	plt.plot(x1,z1,'-o')
	plt.xlabel("Reduced Dimension")
	plt.ylabel("Time (secs)")
	plt.title("Reduced Dimension vs Time")
	plt.show()

	plt.plot(x1,y1,'-o')
	plt.xlabel("Reduced Dimension")
	plt.ylabel("Accuracy")
	plt.title("Reduced Dimension vs Accuracy")
	plt.show()

def plot_knn():
	#############
	#### KNN ####
	#############
	x1=np.array([1,3,5,7,9]) #K
	y1=np.array([84.97,85.41,85.54,85.39,85.19]) #ACCURACY
	z1=np.array([849.688,883.899,824.215,836.036,811.544]) #TIME

	plt.plot(x1,z1,'-o')
	plt.xlabel("K")
	plt.ylabel("Time (secs)")
	plt.title("K vs Time")
	plt.show()

	plt.plot(x1,y1,'-o')
	plt.xlabel("K")
	plt.ylabel("Accuracy")
	plt.title("K vs Accuracy")
	plt.show()

def plot_knn_pca():
	#################
	#### PCA KNN ####
	#################
	x1=np.array([75,80,85,90,95,98]) #PCA VARIANCE
	y1=np.array([80.54,82.66,84.02,85.00,85.21,85.30]) #ACCURACY
	z1=np.array([12.686,13.653,15.383,22.46,42.351,88.585]) #TIME

	x2=np.array([75,80,85,90,95,98]) #PCA VARIANCE
	y2=np.array([81.85,84.11,85.13,85.73,85.92,85.93]) #ACCURACY
	z2=np.array([19.083,21.823,27.927,39.522,74.175,124.229]) #TIME

	x3=np.array([75,80,85,90,95,98]) #PCA VARIANCE
	y3=np.array([82.73,84.58,85.72,86.03,86.23,85.93]) #ACCURACY
	z3=np.array([19.835,25.148,27.308,44.753,76.135,123.548]) #TIME

	x4=np.array([75,80,85,90,95,98]) #PCA VARIANCE
	y4=np.array([83.26,84.93,85.71,86.01,86.19,85.75]) #ACCURACY
	z4=np.array([22.541,23.603,30.750,44.337,93.838,130.235]) #TIME

	x5=np.array([75,80,85,90,95,98]) #PCA VARIANCE
	y5=np.array([83.13,84.78,85.77,86.15,86.02,85.55]) #ACCURACY
	z5=np.array([20.543,24.667,30.972,43.068,75.129,131.803]) #TIME

	plt.plot(x1,z1,'-o')
	plt.plot(x2,z2,'-o')
	plt.plot(x3,z3,'-o')
	plt.plot(x4,z4,'-o')
	plt.plot(x5,z5,'-o')
	plt.xlabel("Explained Variance")
	plt.ylabel("Time (secs)")
	plt.title("Explained Variance vs Time")
	plt.legend(('k1','k3','k5','k7','k9'))
	plt.show()

	plt.plot(x1,y1,'-o')
	plt.plot(x2,y2,'-o')
	plt.plot(x3,y3,'-o')
	plt.plot(x4,y4,'-o')
	plt.plot(x5,y5,'-o')
	plt.xlabel("Explained Variance")
	plt.ylabel("Accuracy")
	plt.title("Explained Variance vs Accuracy")
	plt.legend(('k1','k3','k5','k7','k9'))
	plt.show()

def plot_knn_lda():

	x1=np.array([1,2,3,4,5,6,7,8,9]) #K
	y1=np.array([40.52,51.51,60.17,63.13,69.21,73.30,74.72,78.17,79.11]) #ACCURACY
	z1=np.array([20.023,21.417,23.725,22.314,22.057,24.363,24.276,21.796,23.867]) #TIME

	x2=np.array([1,2,3,4,5,6,7,8,9]) #K
	y2=np.array([42.75,54.90,62.95,66.33,72.49,76.10,77.49,81.0,81.40]) #ACCURACY
	z2=np.array([21.245,20.614,21.238,21.251,20.438,20.605,19.428,20.101,22.577]) #TIME

	x3=np.array([1,2,3,4,5,6,7,8,9]) #K
	y3=np.array([44.19,56.57,65.19,67.92,73.75,77.73,78.60,81.61,82.21]) #ACCURACY
	z3=np.array([19.971,20.328,20.187,24.047,22.949,22.972,24.116,24.439,24.186]) #TIME

	x4=np.array([1,2,3,4,5,6,7,8,9]) #K
	y4=np.array([45.03,57.59,65.63,68.43,74.16,78.30,79.04,81.97,82.79]) #ACCURACY
	z4=np.array([22.699,20.329,24.452,22.805,23.715,24.654,27.111,24.355,22.545]) #TIME

	x5=np.array([1,2,3,4,5,6,7,8,9]) #K
	y5=np.array([45.37,58.10,66.47,68.96,74.47,78.99,79.27,82.22,83.14]) #ACCURACY
	z5=np.array([21.693,24.845,22.196,20.633,20.923,23.779,21.160,21.636,25.063]) #TIME

	plt.plot(x1,z1,'-o')
	plt.plot(x2,z2,'-o')
	plt.plot(x3,z3,'-o')
	plt.plot(x4,z4,'-o')
	plt.plot(x5,z5,'-o')
	plt.xlabel("Reduced Dimension")
	plt.ylabel("Time (secs)")
	plt.title("Reduced Dimension vs Time")
	plt.legend(('k1','k3','k5','k7','k9'))
	plt.show()

	plt.plot(x1,y1,'-o')
	plt.plot(x2,y2,'-o')
	plt.plot(x3,y3,'-o')
	plt.plot(x4,y4,'-o')
	plt.plot(x5,y5,'-o')
	plt.xlabel("Reduced Dimension")
	plt.ylabel("Accuracy")
	plt.title("Reduced Dimension vs Accuracy")
	plt.legend(('k1','k3','k5','k7','k9'))
	plt.show()
	

def main():

	#get_accuracy()
	
	#plot_gauss_pca()
	#plot_gauss_lda()

	#plot_knn()
	#plot_knn_pca()
	#plot_knn_lda()
	
	pass

if __name__=="__main__":
	main()