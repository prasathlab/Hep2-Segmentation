
import numpy as np
import cv2
import random
import pdb


def get_random_patches(imgs, masks, resize_dim, n_patches_per_img):
    img_patches = None
    mask_patches = None
    for img, mask in zip(imgs, masks):
        img_patch, mask_patch = extract_random_patches(img, mask,
                                                       resize_dim[0], resize_dim[1],
                                                       n_patches_per_img
                                                       )
        if img_patches is None and mask_patches is None:
            img_patches = img_patch
            mask_patches = mask_patch
        else:
            img_patches = np.concatenate([img_patches, img_patch], axis=0)
            mask_patches = np.concatenate([mask_patches, mask_patch], axis=0)
    return img_patches, mask_patches


def extract_random_patches(img, mask, patch_h, patch_w, patch_per_img):
    img_h = img.shape[0]  # height of the full image
    img_w = img.shape[1]  # width of the full image
    img_patches, mask_patches = [], []
    k = 0
    while k < patch_per_img:
        x_center = random.randint(0 + int(patch_w / 2), img_w - int(patch_w / 2))
        y_center = random.randint(0 + int(patch_h / 2), img_h - int(patch_h / 2))
        img_patch = img[y_center - int(patch_h / 2): y_center + int(patch_h / 2),
                    x_center - int(patch_w / 2) : x_center + int(patch_w / 2),
                    :]
        img_patches.append(img_patch)
        mask_patch = mask[y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                     x_center - int(patch_w / 2):x_center + int(patch_w / 2),
                     :]
        mask_patches.append(mask_patch)
        k += 1
    return np.array(img_patches), np.array(mask_patches)

def mask_binarization(masks):
    masks = masks / 255
    masks = masks.astype("uint8")
    # Binarize the masks
    n_els = masks.size
    zero_els = np.count_nonzero(masks == 0)
    one_els = np.count_nonzero(masks == 1)
    is_binary = n_els - (zero_els + one_els)
    if is_binary > 0:
        masks = np.where(masks > 0.99, 1, 0)
    return masks

def image_mask_scaling(imgs, masks):
    imgs = imgs / 255.
    masks = mask_binarization(masks)
    return imgs, masks


def resize_images_masks(train_imgs, train_masks, resize_dim):
    resize_imgs, resize_masks = [], []
    for img, mask in zip(train_imgs, train_masks):
        if img.shape[:-1] == resize_dim:
            resize_img = img
            resize_mask = mask
            print("this happened")
        else:
            resize_img = cv2.resize(img, resize_dim, interpolation=cv2.INTER_AREA)
            resize_mask = cv2.resize(mask, resize_dim, interpolation=cv2.INTER_AREA)
        resize_imgs.append(resize_img)
        resize_masks.append(resize_mask)

    resize_imgs = np.array(resize_imgs)
    resize_masks = np.array(resize_masks)
    return resize_imgs, resize_masks

def image_normalization(imgs):
    img_ch_mean = imgs.mean(axis=(0, 1, 2), keepdims=True)
    img_ch_std = imgs.std(axis=(0, 1, 2), keepdims=True)
    imgs = (imgs - img_ch_mean) / img_ch_std
    return (imgs, img_ch_mean, img_ch_std)


def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):

    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[-1] == 1 or full_imgs.shape[-1] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[1]  # height of the full image
    img_w = full_imgs.shape[2]  # width of the full image
    leftover_h = (img_h - patch_h) % stride_h  # leftover on the h dim
    leftover_w = (img_w - patch_w) % stride_w  # leftover on the w dim

    if (leftover_h != 0):  # change dimension of img_h

        # print("\nthe side H is not compatible with the selected stride of " + str(stride_h))
        # print("img_h " + str(img_h) + ", patch_h " + str(patch_h) + ", stride_h " + str(stride_h))
        # print("(img_h - patch_h) MOD stride_h: " + str(leftover_h))
        # print("So the H dim will be padded with additional " + str(stride_h - leftover_h) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0],
                                  img_h + (stride_h - leftover_h),
                                  img_w,
                                  full_imgs.shape[-1])
                                 )
        tmp_full_imgs[0:full_imgs.shape[0], 0:img_h, 0:img_w, 0:full_imgs.shape[-1]] = full_imgs
        full_imgs = tmp_full_imgs

    if (leftover_w != 0):  # change dimension of img_w

        # print("the side W is not compatible with the selected stride of " + str(stride_w))
        # print("img_w " + str(img_w) + ", patch_w " + str(patch_w) + ", stride_w " + str(stride_w))
        # print("(img_w - patch_w) MOD stride_w: " + str(leftover_w))
        # print("So the W dim will be padded with additional " + str(stride_w - leftover_w) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0],
                                  full_imgs.shape[1],
                                  img_w + (stride_w - leftover_w),
                                  full_imgs.shape[-1])
                                 )
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:img_w, 0:full_imgs.shape[-1]] = full_imgs
        full_imgs = tmp_full_imgs

    #print("new full images shape: \n" + str(full_imgs.shape))
    return full_imgs


# Divide all the full_imgs in pacthes
def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[-1] == 1 or full_imgs.shape[-1] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[1]  # height of the full image
    img_w = full_imgs.shape[2]  # width of the full image
    assert ((img_h - patch_h) % stride_h == 0 and (img_w - patch_w) % stride_w == 0)
    N_patches_img = ((img_h - patch_h) // stride_h + 1) * ((img_w - patch_w) // stride_w + 1)  # // --> division between integers
    N_patches_tot = N_patches_img * full_imgs.shape[0]

    # print("Number of patches on h : " + str(((img_h - patch_h) // stride_h + 1)))
    # print("Number of patches on w : " + str(((img_w - patch_w) // stride_w + 1)))
    # print("number of patches per image: " + str(N_patches_img) + ", totally for this dataset: " + str(N_patches_tot))
    patches = np.empty((N_patches_tot, patch_h, patch_w, full_imgs.shape[-1]))
    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                patch = full_imgs[i, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w, :]
                patches[iter_tot] = patch
                iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    return patches  # array with all the full_imgs divided in patches


def recompose_overlap(preds, img_h, img_w, stride_h, stride_w):

    assert (len(preds.shape)==4)  #4D arrays
    assert (preds.shape[-1]==1 or preds.shape[-1]==3)  #check the channel is 1 or 3
    patch_h = preds.shape[1]
    patch_w = preds.shape[2]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    # print ("N_patches_h: " +str(N_patches_h))
    # print ("N_patches_w: " +str(N_patches_w))
    # print ("N_patches_img: " +str(N_patches_img))
    assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img

   # print ("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")

    full_prob = np.zeros((N_full_imgs, img_h, img_w, preds.shape[-1]))  #itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs, img_h, img_w, preds.shape[-1]))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i, h*stride_h:(h*stride_h)+patch_h, w*stride_w:(w*stride_w)+patch_w, :] += preds[k]
                full_sum[i, h*stride_h:(h*stride_h)+patch_h, w*stride_w:(w*stride_w)+patch_w, :] += 1
                k += 1

    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0)  #at least one
    final_avg = full_prob/full_sum
    # print (final_avg.shape)

    assert(np.max(final_avg)<=1.0) #max value for a pixel is 1.0
    assert(np.min(final_avg)>=0.0) #min value for a pixel is 0.0

    return final_avg

