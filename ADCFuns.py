import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.gaussian_process import GaussianProcessClassifier

def dist(p1, p2):
	return np.linalg.norm(p1-p2)

def regionQuery(data, avgDen, p1, RT, DT):
	result1 = set([])
	result2 = set([])
	for p in range(data.shape[0]):
		if dist(data[p], data[p1]) < RT and avgDen[p] > DT:
			result1.add(p)
		if dist(data[p], data[p1]) < RT and avgDen[p] < -DT:
			result2.add(p)
	return result1, result2

def mergeCluster1(p1, c, data, neighborPts, visited, cluster, avgDen, RT, DT):
	cluster[p1] = c
	i = 0
	while(i < len(neighborPts)):
		p = sorted(list(neighborPts))[i]
		if visited[p] == 0:
			visited[p] = 1
			neighborPts1 = regionQuery(data, avgDen, p, RT, DT)
			if len(neighborPts1) > DT:
				neighborPts = neighborPts.union(neighborPts1)
				i = 0
		if cluster[p] == 0:
			cluster[p] = c
		i += 1
	return

def mergeCluster2(p1, c, data, neighborPts, visited, cluster, avgDen, RT, DT):
	cluster[p1] = c
	i = 0
	while(i < len(neighborPts)):
		p = sorted(list(neighborPts))[i]
		if visited[p] == 0:
			visited[p] = 1
			neighborPts1 = regionQuery(data, avgDen, p, RT, DT)
			if len(neighborPts1) < -DT:
				neighborPts = neighborPts.union(neighborPts1)
				i = 0
		if cluster[p] == 0:
			cluster[p] = c
		i += 1
	return



def grid_algorithm(data, m, sample,sample_label, kPar):
# STEP 1: divide into m parts
	maxDim = np.array([max(data[:,d]) for d in range(data.shape[1])])
	minDim = np.array([min(data[:,d]) for d in range(data.shape[1])])
	gridDim = (maxDim - minDim) / m
	gridLabel = [(lambda point: [int(
	 	(point[d]-minDim[d])/gridDim[d]) for d in range(data.shape[1])])(point) for point in data]
# STEP 2: calculate RT
	gridDict = {}
	for p in range(data.shape[0]):
		if str(gridLabel[p]) not in gridDict:
			gridDict[str(gridLabel[p])] = [p]
		else:
			gridDict[str(gridLabel[p])].append(p)

	avgDist = np.zeros(data.shape[0])
	
	for i in range(data.shape[0]):
		neighbor = []
		for d in range(data.shape[1]):
			if d > 0:
				target = [gridLabel[i][x] + ([0] * d + [-1] + [0] * (data.shape[1] - 1 - d))[x] for x in range(data.shape[1])]
				if str(target) in gridDict:
					neighbor += gridDict[str(target)]
			if d < data.shape[1] - 1:
				target = [gridLabel[i][x] + ([0] * d + [1] + [0] * (data.shape[1] - 1 - d))[x] for x in range(data.shape[1])]
				if str(target) in gridDict:
					neighbor += gridDict[str(target)]
		if len(neighbor) > 0:
			avgDist[i] = np.mean([dist(data[i], data[x]) for x in neighbor])
	RT = 0.3

# STEP 3: calculate DT
	avgCellDen = {}
	avgDen = np.zeros(data.shape[0])
	for key in gridDict:
		cellDen = []
		for p in gridDict[key]:
			neighbor = []
			target = gridLabel[p]
			neighbor += gridDict[str(target)]
			for d in range(data.shape[1]):
				if d > 0:
					target = [gridLabel[p][x] + ([0] * d + [-1] + [0] * (data.shape[1] - 1 - d))[x] for x in range(data.shape[1])]
					if str(target) in gridDict:
						neighbor += gridDict[str(target)]
				if d < data.shape[1] - 1:
					target = [gridLabel[i][x] + ([0] * d + [1] + [0] * (data.shape[1] - 1 - d))[x] for x in range(data.shape[1])]
					if str(target) in gridDict:
						neighbor += gridDict[str(target)]
			avgDen[p] = sum([dist(data[p], data[x]) < RT for x in neighbor])
			cellDen.append(avgDen[p])
		if len(cellDen) > 0:
			avgCellDen[key] = np.mean(cellDen)

	DT = 4

# STEP 3.5: Kriging
	clf = GaussianProcessClassifier(warm_start=True).fit(sample, sample_label)
	labelProb = clf.predict_proba(data)[:,1]
	weight = kPar * labelProb - kPar / 2
	weightedDen = np.array([weight[i]*avgDen[i] for i in range(len(avgDen))])


# STEP 4: Clustering
	cluster = np.zeros(data.shape[0])

	c = 0
	visited = np.zeros(data.shape[0])
	for p in range(data.shape[0]):
		if visited[p] == 1:
			continue
		visited[p] = 1
		neighborPts1, neighborPts2 = regionQuery(data, avgDen, p, RT, DT)
		c += 1
		mergeCluster1(p, c, data, neighborPts1, visited, cluster, weightedDen, RT, DT)
		mergeCluster2(p, c, data, neighborPts2, visited, cluster, weightedDen, RT, DT)
	if kPar != 0:
		db = DBSCAN(eps=0.3, min_samples=m).fit(data,  sample_weight= weight)
	else:
		db = DBSCAN(eps=0.3, min_samples=m).fit(data)
	return db




def ADClust(X, m,sample,sample_label, kPar):
	db1 = grid_algorithm(X,m,sample,sample_label, kPar)
	db2 = grid_algorithm(X,m,sample,sample_label, -kPar)
	core_samples_mask1 = np.zeros_like(db1.labels_, dtype=bool)
	core_samples_mask1[db1.core_sample_indices_] = True
	core_samples_mask2 = np.zeros_like(db2.labels_, dtype=bool)
	core_samples_mask2[db2.core_sample_indices_] = True

	non_clustered_weight = [1-(core_samples_mask1[i] or core_samples_mask2[i]) for i in range(len(core_samples_mask2))]
	non_X = [X[i] for i in range(len(non_clustered_weight)) if non_clustered_weight[i] == True]
	non_X_index = [i for i in range(len(non_clustered_weight)) if non_clustered_weight[i] == True]

	db3 =  grid_algorithm(np.array(non_X),m,sample,sample_label, 0)
	core_samples_mask3_F = np.zeros_like(db3.labels_, dtype=bool)
	core_samples_mask3_F[db3.core_sample_indices_] = True

	core_samples_mask3 = np.zeros(len(X))
	for i in range(len(non_X_index)):
		core_samples_mask3[non_X_index[i]] = core_samples_mask3_F[i]

	db4 = grid_algorithm(X,m,sample,sample_label,0)
	core_samples_mask4_F = np.zeros_like(db4.labels_, dtype=bool)
	core_samples_mask4_F[db4.core_sample_indices_] = True

	core_samples_mask4 = np.zeros(len(X))
	for i in range(len(X)):
		core_samples_mask4[i] = core_samples_mask4_F[i]

	mixlabel = []
	unknownlabel = []
	for index in non_X_index:
		if db4.labels_[index] > -1:
			nonlabel = db4.labels_[index]
			if nonlabel in mixlabel:
				continue
			elif nonlabel in unknownlabel:
				continue
			else:
				unknownflag = 1
				for i in range(len(core_samples_mask4)):
					if db4.labels_[i] == nonlabel:
						if core_samples_mask1[i] or core_samples_mask2[i]:
							mixlabel.append(nonlabel)
							unknownflag = 0
							break
				if unknownflag:
					unknownlabel.append(nonlabel)

	core_samples_mask4 = np.zeros(len(X))
	for i in range(len(non_X_index)):
		if (db4.labels_[non_X_index[i]] in unknownlabel) and core_samples_mask3[non_X_index[i]]:
			core_samples_mask4[non_X_index[i]] = 1

	labels_db = core_samples_mask1 + 2 * core_samples_mask2 + 3 * core_samples_mask3 + core_samples_mask4
	return labels_db

