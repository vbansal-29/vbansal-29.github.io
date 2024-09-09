# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images
import numpy as np
import skimage as sk
import skimage.io as skio
import scipy
from skimage.transform import resize
from skimage.metrics import structural_similarity
import os

# gets list of all images (.jpg and .tif) files in the data folder
images = [f for f in os.listdir("../data") if f.split(".")[1] in set(['png', 'tif', 'jpg'])]

# loops through each image
for image in images:
    imname = os.path.join("../data", image)
    print(imname)
    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # align the images
    # functions that might be useful for aligning the images include:
    # np.roll, np.sum, sk.transform.rescale (for multiscale)

    # function to crop images by 10% on each border
    def crop(image):
        h = int(image.shape[0] * 10/100)
        w = int(image.shape[1] * 10/100)

        return image[h:-h, w:-w]
    
    # euclidean distance cost function 
    def ssd(image, reference):
        return np.sqrt(np.sum((image - reference) ** 2))

    # normalized cross correlation (ncc) cost function
    def ncc(image, reference):
        image_flat = image.flatten()
        ref_flat = reference.flatten()

        image_flat = image_flat - np.mean(image_flat)
        ref_flat = ref_flat - np.mean(ref_flat)

        image_norm = np.sqrt(np.sum(image_flat ** 2))
        ref_norm = np.sqrt(np.sum(ref_flat ** 2))
        return np.dot(image_flat, ref_flat)/(image_norm * ref_norm)

    # scale the image down by a factor of 1/2
    def scale(image):
        new_shape = (image.shape[0] // 2, image.shape[1] // 2)
        return resize(image, new_shape, anti_aliasing=True)
    
    # search over a predefined range of shifts and classify each using a similarity metric to find the best shift
    def align(image, reference, search_range=15):
        
        # initialize basecases
        best_shift = (0, 0)
        best_image = image
        cropped_image = crop(image)
        cropped_reference = crop(reference)
        metric = structural_similarity(cropped_image, cropped_reference, data_range=1.0)
        
        # loop through each case
        for rshift in range(-search_range, search_range):
            for cshift in range(-search_range, search_range):
                shifted_image = np.roll(image , (rshift, cshift), axis=(0, 1))
                cropped_image = crop(shifted_image)
                similarity = structural_similarity(cropped_image, cropped_reference, data_range=1.0)
                
                if similarity > metric:
                    best_shift = (rshift, cshift)
                    best_image = shifted_image
                    metric = similarity
        
        # return the best_image and best row and col shifts
        return best_image, best_shift
    
    # image pyramid implementation for larger res .tif files 
    def pyramid(image, reference, level):

        # recursively scale down image until (3000, 3000) file size roughly reduces to (400, 400)
        if level == 0:
            print(image.shape)
            return align(image, reference, 30)[1]
        else:
            image_scale = scale(image)
            ref_scale = scale(reference)
            
            rshift, cshift = pyramid(image_scale, ref_scale, level - 1)

            # guess level's shifts by multiplying level - 1's shifts by 2
            rshift *= 2
            cshift *= 2
            
            return rshift, cshift
                    
    level = 0 if imname.split('.')[2] == 'jpg' else 3

    # align g with b
    rshift, cshift = pyramid(g, b, level)
    ag = np.roll(g , (rshift, cshift), axis=(0, 1))
    print(rshift, cshift)
    
    # align r with b
    rshift, cshift = pyramid(r, b, level)
    ar = np.roll(r , (rshift, cshift), axis=(0, 1))
    print(rshift, cshift)

    # create a color image by stacking r, g, & b
    im_out = np.dstack([ar, ag, b])
    im_out = sk.img_as_ubyte(im_out)

    # save the image
    fname = f"./outputs/{imname.split('/')[-1]}"
    skio.imsave(fname, im_out)

    # display the image
    skio.imshow(im_out)
    skio.show()