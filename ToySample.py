import numpy as np
import matplotlib.pyplot as plt
import ADCFuns

# ABA
kPar = 25
np.random.seed(1)
X = np.random.rand(2000, 2) * 6 - 3

sample = X[1:100,]
sample_label = []
sampleN = []
sampleA = []

for i in range(len(sample)):
	if sample[i,0] < 1 and sample[i,0] > -1:
		sample_label.append(1)
		sampleN.append(sample[i])
	else:
		sample_label.append(-1)
		sampleA.append(sample[i])

abnormalN1 = 19
abnormalN2 = 24
normalN = 5
sampleA = sampleA[abnormalN1:abnormalN2]
sampleN = sampleN[:normalN]
sample = np.vstack((sampleA,sampleN))
sample_label = [-1] * (abnormalN2-abnormalN1) + [1] * normalN



labels_db = ADCFuns.ADClust(X,10,sample,sample_label, kPar)


unique_labels_db = set(labels_db)
colors_db = ['black','blue','red','purple','yellow']
colors_db1 = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels_db)))


print("Make A Plot")
fig, ax = plt.subplots()

# Draw bound
# ABA
plt.plot([-1,-1], [-3,3], 'k-', lw=2)
plt.plot([1,1], [-3,3], 'k-', lw=2)


for k, col in zip(unique_labels_db, colors_db):
    if k == 0:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels_db == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor=col, markersize=6, alpha = 0.3)

sampleNormal = [x for x in range(len(sample_label)) if sample_label[x] == 1]
sampleAbnormal = [x for x in range(len(sample_label)) if sample_label[x] == -1]

plt.plot(sample[sampleNormal, 0], sample[sampleNormal, 1], 'h', markerfacecolor='b',
		markeredgecolor='k', markersize=12)
plt.plot(sample[sampleAbnormal, 0], sample[sampleAbnormal, 1], 'h', markerfacecolor= 'r',
		markeredgecolor='k', markersize=12)
plt.show()