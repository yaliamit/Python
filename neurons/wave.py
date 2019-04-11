from scipy.io import loadmat
from sklearn.cluster import KMeans
import copy
import pylab as py
import numpy as np
data=loadmat('data_20161214_N30.mat')
M=np.zeros((4,56))
adata=data['data']
n4=adata[0,0]['NEURO']['Waveforms'][0,0]['wf_sig004a_wf'][0,0]
M[0,:]=np.mean(n4,axis=0)
n19=adata[0,0]['NEURO']['Waveforms'][0,0]['wf_sig019a_wf'][0,0]
M[1,:]=np.mean(n19,axis=0)
n20=adata[0,0]['NEURO']['Waveforms'][0,0]['wf_sig020a_wf'][0,0]
M[2,:]=np.mean(n20,axis=0)
n21=adata[0,0]['NEURO']['Waveforms'][0,0]['wf_sig021a_wf'][0,0]
M[3,:]=np.mean(n21,axis=0)
N=np.vstack((n19,n20))
ii=np.int64(range(N.shape[0]))
np.random.shuffle(ii)
Ni=np.float32(N[ii])
num_clusters=4
kmeans=KMeans(n_clusters=num_clusters,random_state=123).fit(Ni)
yi=kmeans.predict(Ni)
y=kmeans.predict(N)
correct0=(np.sum(y[0:n19.shape[0]]==0)+np.sum(y[n19.shape[0]:,]==1))
correct1=(np.sum(y[0:n19.shape[0]]==1)+np.sum(y[n19.shape[0]:,]==0))
correct=np.float32(np.maximum(correct0,correct1))/N.shape[0]
ii19=[]
t=0
while len(ii19) < 10:
    if (ii[t]<n19.shape[0]):
        ii19.append(ii[t])
    t=t+1
ii19=np.array(ii19)
ii20=[]
t=0
while len(ii20) < 10:
    if (ii[t]>=n19.shape[0]):
        ii20.append(ii[t]-n19.shape[0])
    t=t+1

ii20=np.array(ii20)


for j in range(10):
    py.plot(n19[ii19[j],:])
    py.plot(n20[ii20[j],:])
py.show()



for i in range(num_clusters):
    py.plot(kmeans.cluster_centers_[i,:])
py.show()

cols=['red','blue','green','cyan']

for i in range(num_clusters):
    iy=np.int32(np.where(y==i)[0])
    for j in range(10):
        py.plot(N[iy[j],:],color=cols[i])
    py.show()
# py.subplot(1,2,1)
# for j in range(10):
#     py.plot(n19[ii19[j],:],color=cols[y[ii19[j]]])
# py.subplot(1,2,2)
# for j in range(10):
#     py.plot(n20[ii20[j],:],color=cols[y[ii20[j]+n19.shape[0]]])
# py.show()

