import skimage.color
import skimage.io
import numpy as np
import matplotlib.pyplot as plt


def addNoise(image, sigma):

    if image.ndim == 2:

        n = np.random.normal(0, sigma, image.shape)
        img_n = np.float32(image) + n
        img_n = np.float32(img_n)
        img_n[img_n>255.0] = 255.0
        img_n[img_n<0] = 0
        return img_n

    elif image.ndim == 3:

        # Image with alpha channel
        if image.shape[-1] == 4:
            image = image[..., :-1]

        ycbcr = skimage.color.rgb2ycbcr(image)
        img = ycbcr[:,:,0]
        n = np.random.normal(0, sigma, img.shape)
        img_n = img + n
        img_n[img_n>255.0] = 255.0
        img_n[img_n<0] = 0
        ycbcr[:,:,0] = img_n
        img_c = skimage.color.ycbcr2rgb(ycbcr)
        img_c[img_c>1.0] = 1.0
        img_c[img_c<0] = 0
        img_c = np.float32(img_c)
        return img_c


def mse_psnr(image_ref, image):

    if image_ref.ndim == 2 and image.ndim == 2:
        scale = 255
        scale_ref = 255

        if np.max(image)>1:
            scale = 1
        if np.max(image_ref)>1:
            scale_ref = 1

        mse = np.mean(np.square(image*scale - np.float32(image_ref)*scale_ref))

    elif image_ref.ndim == 3 and image.ndim == 3:
        g_ref = skimage.color.rgb2gray(image_ref)
        g_o = skimage.color.rgb2gray(image)
        mse = 255*255*np.mean(np.square(g_ref - g_o))

    psnr = 10*np.log10(255*255/mse)

    return mse, psnr


def visualize_image(img_n, im_den, psnr_n, psnr_d):
    """
    Visualize before and after images.

    img_n: noisy image
    im_den: denoised image
    psnr_n: pnsr of noisy image
    psnr_d : pnsr of denoised image
    """
    plt.figure(figsize=(14, 14))
    plt.subplot(1, 2, 1)
    plt.title("Noisy image PSNR = %2.2f dB" % psnr_n, fontsize=10)
    plt.axis('off')
    if img_n.ndim == 2:
        if np.max(img_n) > 1:
            img_n = img_n / 255.0
        plt.imshow(skimage.color.gray2rgb(img_n))
    else:
        plt.imshow(img_n)

    plt.subplot(1, 2, 2)
    plt.title("Denoised image PSNR = %2.2f dB" % psnr_d, fontsize=10)
    plt.axis('off')
    if img_n.ndim == 2:
        if np.max(img_n) > 1:
            im_den = im_den / 255.0
        plt.imshow(skimage.color.gray2rgb(im_den))
    else:
        plt.imshow(im_den)

    plt.show()
