import albumentations as A
import cv2
from PIL import Image
import numpy as np
import os

if not os.path.exists('aug'): #directory to save augmented images
    os.makedirs('aug')

if not os.path.exists('aug//A'):
    os.makedirs('aug//A')

if not os.path.exists('aug//B'):
    os.makedirs('aug//B')

if not os.path.exists('aug//C'):
    os.makedirs('aug//C')

if not os.path.exists('aug//D'):
    os.makedirs('aug//D')

if not os.path.exists('aug//E'):
    os.makedirs('aug//E')

if not os.path.exists('aug//F'):
    os.makedirs('aug//F')


IMG_WIDTH, IMG_HEIGHT, AUG_IMG_NUM = 299, 299, 5 #AUG_IMG_NUM: generate AUG_IMG_NUM examples for each image
images = []
in_a, out_a = os.path.join('images', '1'), os.path.join('aug', 'A')
in_b, out_b = os.path.join('images', '2'), os.path.join('aug', 'B')
in_c, out_c = os.path.join('images', '3'), os.path.join('aug', 'C')
in_d, out_d = os.path.join('images', '4'), os.path.join('aug', 'D')
in_e, out_e = os.path.join('images', '5'), os.path.join('aug', 'E')
in_f, out_f = os.path.join('images', '6'), os.path.join('aug', 'F')


dirs_arr = [(in_a, out_a), 
            (in_b, out_b),
            (in_c, out_c),
            (in_d, out_d),
            (in_e, out_e),
            (in_f, out_f)]

aug_im_counter = 0

transform = A.Compose(
    [
        A.Resize(width=IMG_WIDTH, height=IMG_HEIGHT),
        A.Rotate(p=0.3, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=1),
        A.RGBShift(25, 25, 25, p=0.9),
        A.Blur(blur_limit=3, p=0.6),
        A.RandomShadow(p=0.5),
        A.RandomFog(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
        A.RandomRain(p=0.5),
    ]
)
print("Defined transformations....")

def augment(input_dir, output_dir):
    print('Performing augmentation...')
    aug_im_counter = 1

    for im in os.listdir(input_dir):
        image = Image.open(os.path.join(input_dir, im))
        image = np.array(image)

        for _ in range(AUG_IMG_NUM):
            output_im = os.path.join(output_dir, 'augmented_image_' + str(aug_im_counter) + '.jpg')

            if not os.path.exists(output_im):
                
                try:  #skip the images that will cause any problem and go to the next one.
                    augmentations = transform(image=image)
                except:
                    print('Error!')
                    break

                aug_im = augmentations['image']
                augmented_image = Image.fromarray(aug_im)
                augmented_image.save(output_im)

            print(f'Image {aug_im_counter} done...')
            aug_im_counter += 1    

for in_dir, out_dir in dirs_arr:
    augment(in_dir, out_dir)