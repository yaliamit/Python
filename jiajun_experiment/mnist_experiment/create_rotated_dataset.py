from CNNForMnist import build_cnn, load_data
import numpy as np
import cv2

def rotateImage(image, angle):
    if len(image.shape) == 3:
            image = image[0]
    image_center = tuple(np.array(image.shape)/2)
    rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
    return np.array(result[np.newaxis, :, :], dtype = np.float32)

def extend_image(inputs, size = 40):
    extended_images = np.zeros((inputs.shape[0], 1, size, size), dtype = np.float32)
    margin_size = (40 - inputs.shape[2]) / 2
    extended_images[:, :, margin_size:margin_size + inputs.shape[2], margin_size:margin_size + inputs
.shape[3]] = inputs
    return extended_images





X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")
X_test = extend_image(X_test, 40)

test_size = y_test.shape[0]
all_images = []
all_labels = []
for j in range(5):
    angles_1 = list(np.random.randint(low = -20, high = -5, size = test_size // 2))
    angles_2 = list(np.random.randint(low = 5, high = 20, size = test_size // 2))
    angles = np.array(angles_1 + angles_2)
    np.random.shuffle(angles)
    rotated_image = np.array([rotateImage(X_test[i], angles[i]) for i in range(test_size)], dtype = np.float32)
    all_images.append(rotated_image)
    all_labels.append(y_test)
all_images = np.vstack(all_images)
all_labels = np.hstack(all_labels)

print(all_images.shape, all_labels.shape)

index = np.arange(5 * test_size)
np.random.shuffle(index)

all_images = all_images[index, 0, 6: 34, 6:34]
all_labels = all_labels[index]


np.save("/home/jiajun/.mnist/X_test_rotated.npy", all_images)
np.save("/home/jiajun/.mnist/Y_test_rotated.npy", all_labels)

