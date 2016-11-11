import numpy as np
import amitgroup as ag
from sklearn import svm
import os
ag.set_verbose(True)

assert os.environ.get('MNIST'), "Please set the environment variable MNIST to your MNIST directory"

# Number of training samples per class
N = 100

# Number of testing samples
N_TEST = 1000

bedges_settings = dict(contrast_insensitive=False, spread='orthogonal', radius=2)
part_size = (5, 5)
num_parts = 100

descriptor = ag.features.PartsDescriptor(part_size, num_parts, edges_threshold=25, 
                                    discard_threshold=1.0, 
                                    bedges_settings=bedges_settings)

settings = dict(spread_radii=(2, 2))

# Load data to get train the parts on
anydata = ag.io.load_mnist('training', selection=slice(2000, 3000), return_labels=False)
descriptor.train_from_images(anydata)

# Load data
X = None#np.zeros((10, N,) + part_size)
labels = np.zeros((10, N))

for d in range(10):
    ims = ag.io.load_mnist('training', [d], selection=slice(N), return_labels=False)
    parts = descriptor.extract_features_many(ims, settings)
    if X is None:
        X = np.zeros((10,) + parts.shape)
    X[d] = parts
    labels[d] = d

flat_X = X.reshape((10*N, -1))
flat_labels = labels.flatten()

# Train an SVM
classifier = svm.SVC(kernel='linear')
classifier.fit(flat_X, flat_labels)

training_success_rate = (classifier.predict(flat_X) == flat_labels).mean()


# Load testing data
ims, test_labels = ag.io.load_mnist('testing', selection=slice(N_TEST))
X_test = descriptor.extract_features_many(ims, settings)
flat_X_test = X_test.reshape((X_test.shape[0], -1))

testing_success_rate = (classifier.predict(flat_X_test) == test_labels).mean()

print 'Training success rate: {0:.2f}'.format(training_success_rate * 100)
print 'Testing success rate:  {0:.2f}'.format(testing_success_rate * 100)
