=================================
FILE DESCRIPTIONS
=================================

I) CODE FOLDER

	1)init.py - Code for implementing Bayes Classifier
	2)lda.py - Code for implementing LDA on Bayes Classifier and KNN
	3)nn.py - Code for implementing K Nearest Neighbor
	4)pca.py - Code for implementing PCA on Bayes Classifier and KNN 
	5)tester.py - Code for plotting results, load individual datafiles

II) DATADUMPS FOLDER

	1) Gauss Folder - contains Bayes Classifier data (3 Files)
		1.1) cov_list.pickle - array of covariance of all classes 
		1.2) final_list.pickle - array of final predictions and ground truth values of shape (10000,2) for Bayes Classifier
		1.3) mean_list.pickle - array of means of all classes
	2) KNN Folder - contains KNN data (5 Files)
		2.1) nn1_preds.pickle - array of final predictions and ground truth values of shape (10000,2) for 1 Nearest Neighbor
		2.2) nn3_preds.pickle - array of final predictions and ground truth values of shape (10000,2) for 3 Nearest Neighbor
		2.3) nn5_preds.pickle - array of final predictions and ground truth values of shape (10000,2) for 5 Nearest Neighbor
		2.4) nn7_preds.pickle - array of final predictions and ground truth values of shape (10000,2) for 7 Nearest Neighbor
		2.5) nn9_preds.pickle - array of final predictions and ground truth values of shape (10000,2) for 9 Nearest Neighbor
	3) LDA Folder - contains LDA data
		3.1) Gauss Folder - contains Bayes Classifier data (9 Files)
			pickle files - array of final predictions and ground truth values of shape (10000,2) for Bayes Classifier after applying LDA of Reduced Dimensions = 1,2,3..9
		3.2) NN Folder - contains KNN Classifier data
			3.2.1) NN1 Folder - contains 1 Nearest Neighbor data (9 Files)
				pickle files - array of final predictions and ground truth values of shape (10000,2) for 1 NN Classifier after applying LDA of Reduced Dimensions = 1,2,3..9
			3.2.2) NN3 Folder - contains 3 Nearest Neighbor data (9 Files)
				pickle files - array of final predictions and ground truth values of shape (10000,2) for 3 NN Classifier after applying LDA of Reduced Dimensions = 1,2,3..9
			3.2.3) NN5 Folder - contains 5 Nearest Neighbor data (9 Files)
				pickle files - array of final predictions and ground truth values of shape (10000,2) for 5 NN Classifier after applying LDA of Reduced Dimensions = 1,2,3..9
			3.2.4) NN7 Folder - contains 7 Nearest Neighbor data (9 Files)
				pickle files - array of final predictions and ground truth values of shape (10000,2) for 7 NN Classifier after applying LDA of Reduced Dimensions = 1,2,3..9
			3.2.5) NN9 Folder - contains 9 Nearest Neighbor data (9 Files)
				pickle files - array of final predictions and ground truth values of shape (10000,2) for 9 NN Classifier after applying LDA of Reduced Dimensions = 1,2,3..9

	4) OCA Folder - contains PCA data
		4.1) Gauss Folder - contains Bayes Classifier data (6 Files)
			pickle files - array of final predictions and ground truth values of shape (10000,2) for Bayes Classifier after applying PCA of Explained Variance = 0.75,0.8,0.85,0.9,0.95,0.98

		4.2) NN Folder - contains KNN Classifier data
			4.2.1) NN1 Folder - contains 1 Nearest Neighbor data (6 Files)
				pickle files - array of final predictions and ground truth values of shape (10000,2) for 1 NN Classifier after applying PCA of Explained Variance = 0.75,0.8,0.85,0.9,0.95,0.98
			4.2.2) NN3 Folder - contains 3 Nearest Neighbor data (6 Files)
				pickle files - array of final predictions and ground truth values of shape (10000,2) for 3 NN Classifier after applying PCA of Explained Variance = 0.75,0.8,0.85,0.9,0.95,0.98
			4.2.3) NN5 Folder - contains 5 Nearest Neighbor data (6 Files)
				pickle files - array of final predictions and ground truth values of shape (10000,2) for 5 NN Classifier after applying PCA of Explained Variance = 0.75,0.8,0.85,0.9,0.95,0.98
			4.2.4) NN7 Folder - contains 7 Nearest Neighbor data (6 Files)
				pickle files - array of final predictions and ground truth values of shape (10000,2) for 7 NN Classifier after applying PCA of Explained Variance = 0.75,0.8,0.85,0.9,0.95,0.98
			4.2.5) NN9 Folder - contains 9 Nearest Neighbor data (6 Files)
				pickle files - array of final predictions and ground truth values of shape (10000,2) for 9 NN Classifier after applying PCA of Explained Variance = 0.75,0.8,0.85,0.9,0.95,0.98

III) FASHION_MNIST FOLDER - Downloaded directly from Fashion-mnist Dataset page https://github.com/zalandoresearch/fashion-mnist


IV) REPORT FOLDER - contains Report.pdf which is the report file  (1 File)


=================================
Libraries Used
=================================

1) Numpy
2) matplotlib 
3) scipy
4) sklearn
5) Pickle
6) math
7) time

=================================
RUN INSTRUCTION
=================================

1) Make sure directory structure is maintained 
2) Make sure all the libraries are installed
3) RUN init.py for implementing Bayes Classifier
4) RUN nn.py for implementing KNN

	###OPTIONAL## Change n_neighbors on line 15 for diff nearest neighbor (1,3,5,7,9)

5)RUN lda.py for implementing LDA on Bayes Classifier and KNN
	5.1) Uncomment lines 92 and 93 for Bayes Classifier
	5.2) Uncomment lines 95 and 96 for KNN

	###OPTIONAL## Change n_components on line 12 for diff reduced dimension (1,2,..9)
	###OPTIONAL## Change n_neighbors on line 19 for diff nearest neighbor (1,3,5,7,9)

6) RUN pca.py for implementing PCA on Bayes Classifier and KNN
	6.1) Uncomment lines 92 and 93 for Bayes Classifier
	6.2) Uncomment lines 95 and 96 for KNN

	###OPTIONAL## Change value on line 12 for diff explained variance (0.0-1.0)
	###OPTIONAL## Change n_neighbors on line 19 for diff nearest neighbor (1,3,5,7,9)

7) RUN tester.py for loading the datadumps and plotting results obtaied
	7.1) Ensure locations of datadumps are correct in get_accuracy function and uncomment as required
		7.1.1) Uncomment line 164 for printing accuracy
	7.2) Uncomment line 166 for plot of PCA on Bayes Classifier
	7.2) Uncomment line 167 for plot of LDA on Bayes Classifier
	7.2) Uncomment line 169 for plot of KNN
	7.2) Uncomment line 170 for plot of PCA on KNN
	7.2) Uncomment line 171 for plot of LDA on KNN
