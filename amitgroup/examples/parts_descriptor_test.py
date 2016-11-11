
import amitgroup as ag
import numpy as np

ag.set_verbose(True)

# This requires you to have the MNIST data set.
data, digits = ag.io.load_mnist('training', selection=slice(0, 100))

pd = ag.features.PartsDescriptor((5, 5), 20, patch_frame=1, edges_threshold=5, samples_per_image=10)

# Use only 100 of the digits
pd.train_from_images(data)

# Save the model to a file. 
#pd.save('parts_model.npy')

# You can then load it again by
#pd = ag.features.PartsDescriptor.load(filename)

# Then you can extract features by
#features = pd.extract_features(image)

# Visualize the parts
ag.plot.images(pd.visparts)
