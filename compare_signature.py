import math
import os
import sys
from PIL import Image
import torch

from Model import SiameseConvNet, distance_metric
from Preprocessing import convert_to_image_tensor, invert_image


def load_img(img_path: str):
    return Image.open(img_path)

def load_data(folder_path: str, filter: str):
    files = os.listdir(folder_path)
    items = [filename for filename in files if filter in filename]
    items.sort()
    return items[0], items[1:]

def load_model():
    device = torch.device('cpu')
    model = SiameseConvNet().eval()
    model.load_state_dict(torch.load('Models/model_large_epoch_20', map_location=device))
    return model


def run_custom():
    args = sys.argv
    if len(args) < 3:
        print(f'invalid arguments')
        return
    
    base_img_path = args[1]
    inpt_img_path = args[2]
    
    base_image = load_img(base_img_path)
    inpt_image = load_img(inpt_img_path)

    base_image_tensor = convert_to_image_tensor(invert_image(base_image)).view(-1, 1, 220, 155)
    inpt_image_tensor = convert_to_image_tensor(invert_image(inpt_image)).view(1, 1, 220, 155)

    model = load_model()

    threshold = 0.145139 # old
    threshold = 0.149314 # new

    mindist = math.inf

    f_A, f_X = model.forward(base_image_tensor, inpt_image_tensor)
    dist = float(distance_metric(f_A, f_X).detach().numpy())
    mindist = min(mindist, dist)

    compare = f'base: {base_img_path}, input: {inpt_img_path}'
    if dist <= threshold:
        print(f'{compare}, match: true, threshold: {threshold}, distance: {mindist}')
    else:
        print(f'{compare}, match: false, threshold: {threshold}, distance: {mindist}')


def run_batch():
    args = sys.argv
    if len(args) < 2:
        print(f'invalid arguments')
        return

    sign = args[1]
    folder_path = './sign_data'
    base_img, input_imgs = load_data(folder_path, sign)

    base_img_path = os.path.join(folder_path, base_img)
    anchor_image = load_img(base_img_path)
    anchor_image_tensor = convert_to_image_tensor(invert_image(anchor_image)).view(-1, 1, 220, 155)
    
    input_images_tensor = []
    for img_name in input_imgs:
        img_path = os.path.join(folder_path, img_name)
        input_image = load_img(img_path)
        input_image_tensor = convert_to_image_tensor(invert_image(input_image)).view(1, 1, 220, 155)
        input_images_tensor.append({'filename': img_name, 'tensor': input_image_tensor})

    model = load_model()

    threshold = 0.145139 # old
    threshold = 0.149314 # new

    for item in input_images_tensor:
        mindist = math.inf

        f_A, f_X = model.forward(anchor_image_tensor, item['tensor'])
        dist = float(distance_metric(f_A, f_X).detach().numpy())
        mindist = min(mindist, dist)

        compare = f'base: {base_img}, input: {item["filename"]}'
        if dist <= threshold:
            print(f'{compare}, match: true, threshold: {threshold}, distance: {mindist}')
        else:
            print(f'{compare}, match: false, threshold: {threshold}, distance: {mindist}')


if __name__ == '__main__':
    # run_batch()
    run_custom()
