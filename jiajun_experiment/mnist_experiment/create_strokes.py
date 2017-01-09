import numpy as np

def generate(images, mega_patch_w, result_sample_w, num_strokes, num_samples):
    image_size, image_channel, image_w, image_h = images.shape
    generated_image = np.zeros((num_samples, image_channel, result_sample_w, result_sample_w))

    for j in range(num_samples):
        t_y = np.random.randint(result_sample_w - mega_patch_w)
        t_x = np.random.randint(result_sample_w - mega_patch_w)
        intensity = 0
        patch = np.zeros((image_channel, mega_patch_w, mega_patch_w))
        while(intensity < 20):
            index = np.random.randint(image_size)
            s_y = np.random.randint(image_w - mega_patch_w)
            s_x = np.random.randint(image_h - mega_patch_w)
            patch = images[index, :, s_x:s_x + mega_patch_w, s_y:s_y + mega_patch_w]
            intensity = np.sum(patch)
        
            
        generated_image[j, :, t_x:t_x + mega_patch_w, t_y:t_y + mega_patch_w] = patch

    return generated_image


def main():
    X = np.load("/home-nfs/jiajun/.mnist/X_train_limited_100.npy")
    result = generate(X, 10, 20, 1, 50)
    print(result.shape)

if __name__ == "__main__":
    main()
    
    
