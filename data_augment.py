import os
import cv2
from tqdm import tqdm
from glob import glob
from albumentations import Transpose, RandomRotate90, HorizontalFlip, VerticalFlip

def load_data(path):
    images = sorted(glob(os.path.join(path, "images/*")))
    masks = sorted(glob(os.path.join(path, "masks/*")))
    return images, masks

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def augment_data(images, masks, save_path, augment=True):
    """
    Function to augment image and mask using albumentation library
    """

    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = x.split("\\")[-1].split(".")
        """ Extracting the name and extension of the image and the mask. """
        image_name = name[0]
        image_extn = name[1]

        name = y.split("\\")[-1].split(".")
        mask_name = name[0]
        mask_extn = name[1]

        """ Reading image and mask. """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        """ Augmentation """
        if augment == True:
            aug = Transpose(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = RandomRotate90(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            save_images = [x, x1, x2, x3, x4]
            save_masks =  [y, y1, y2, y3, y4]

        else:
            save_images = [x]
            save_masks = [y]

        """ Saving the image and mask. """
        idx = 0
        for i, m in zip(save_images, save_masks):

            if len(images) == 1:
                tmp_img_name = f"{image_name}.{image_extn}"
                tmp_mask_name = f"{mask_name}.{mask_extn}"
            else:
                tmp_img_name = f"{image_name}_{idx}.{image_extn}"
                tmp_mask_name = f"{mask_name}_{idx}.{mask_extn}"

            image_path = os.path.join(save_path, "images", tmp_img_name)
            mask_path = os.path.join(save_path, "masks", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1

    return

if __name__ == "__main__":
    BASE_DIR = os.getcwd()
    
    # Loading original images and masks.
    data_path = os.path.join(BASE_DIR, "data")
    images, masks = load_data(data_path)
    print(f"Total Original Images: {len(images)} - Total Original Masks: {len(masks)}")

    # Creating folders.
    create_dir(os.path.join(BASE_DIR, "aug_data/images"))
    create_dir(os.path.join(BASE_DIR, "aug_data/masks"))

    #  Applying data augmentation.
    save_path = os.path.join(BASE_DIR,  "aug_data")

    print("Augmentation in progress...")
    augment_data(images, masks, save_path, augment=True)
    print("Augmentation done!")

    # Loading augmented images and masks.
    images, masks = load_data(save_path)
    print(f"Total Augmented Images: {len(images)} - Total Augmented Masks: {len(masks)}")
