import os
from queue import Queue
from PIL import Image, ImageOps
import numpy as np
import time
from utils import random_bool, make_dir
import matplotlib.pyplot as plt


def generate_transparent_holes(hedge_img, hedge_density=0.3):
    """
    :param hedge_img: PIL Image File
    :param hedge_density: decide how dense is the hedge mask
    :return hedge mask image
    """
    hedge_img = hedge_img.convert('RGBA')  # RGBA (RGB Alpha)
    pixels = hedge_img.getdata()  # convert to ImagingCore object (containing pixel values as tuples)

    # get all the red channel, sort and find the threshold (n13040303 - density)
    r_pixels = [pixel[0] for pixel in pixels]
    r_pixels.sort()
    r_threshold = r_pixels[int((1 - hedge_density) * (len(r_pixels) - 1))]

    # for each pixel if the red channel is smaller than the threshold, replace it with white pixel
    new_pixels = []
    for pixel in pixels:
        if pixel[0] <= r_threshold:
            new_pixels.append((255, 255, 255, 0))
        else:
            new_pixels.append(pixel)

    # place the rearranged pixels into the image object
    hedge_img.putdata(new_pixels)
    return hedge_img


def search_pixel_group(m_img, i, j, width, height, pixel_group_ids, pixel_group_id):
    """
    an algorithm that "spreads" out from pixel (i, j) and check if its neighbor is non-hedge
    if it is non-hedge then they would be allocated into the same group
    :param m_img: image to be searched
    :param i: index i of pixel
    :param j: index j of pixel
    :param width: width of image
    :param height: height of image
    :param pixel_group_ids: list that stores the group ID
    :param pixel_group_id: id for the current pixel group
    :return: area of current pixel group
    """
    pixel_queue = Queue()  # pixels to be processed
    pixel_queue.put((i, j))
    pixel_group_ids[i, j] = pixel_group_id

    area = 0

    while not pixel_queue.empty():
        (i, j) = pixel_queue.get()
        area += 1
        # for all neighbor (all pixels around current pixel)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                # make sure the index is within the scope of image
                if 0 <= i + di < height and 0 <= j + dj < width:
                    if (m_img[i + di, j + dj] > 0) and (pixel_group_ids[i + di, j + dj] == 0):
                        pixel_group_ids[i + di, j + dj] = pixel_group_id
                        # add all neighboring unvisited non-hedge pixel to the queue
                        pixel_queue.put((i + di, j + dj))

    return area


def hedge_denoising(hedge_img, noise_area_threshold=200):
    """
    remove all pixel segments with area smaller than MAX_NOISE_AREA
    :param hedge_img: image to be de-noised
    :param noise_area_threshold: the area threshold for the pixel group to remain in the image
    :return: denoised hedge mask
    """
    width, height = hedge_img.size
    pixels = hedge_img.getdata()
    m_img = np.asarray(pixels)[..., 3].reshape(height, width)

    pixel_group_ids = np.zeros(m_img.shape).astype(int)  # store group ID for each non-hedge pixels
    area = np.zeros(m_img.size).astype(int)  # stores area for each group ID

    pixel_group_id = 0
    new_pixels = []
    for x in range(height):
        for j in range(width):
            # check if current pixel is non-hedge, if it is hedge, then just append
            if m_img[x, j] > 0:
                # if the pixel haven't been visited
                if pixel_group_ids[x, j] == 0:
                    pixel_group_id += 1
                    area[pixel_group_id] = search_pixel_group(m_img, x, j, width, height, pixel_group_ids,
                                                              pixel_group_id)

                # if the current pixel belongs to a group with area smaller than noise_threshold, remove it
                if area[pixel_group_ids[x, j]] < noise_area_threshold:
                    new_pixels.append((255, 255, 255, 0))
                else:
                    new_pixels.append(pixels[len(new_pixels)])
            else:
                new_pixels.append(pixels[len(new_pixels)])

    hedge_img.putdata(new_pixels)
    return hedge_img


def compute_density(hedge_img):
    """ compute the density of the image """
    width, height = hedge_img.size
    pixels = hedge_img.getdata()
    m_img = np.asarray(pixels)[..., 3].reshape(height, width)
    return np.sum(m_img > 0) / m_img.size


def generate_hedge_mask(hedge_img, hedge_density):
    """
    n13040303) first generate holes with algorithm n13040303
    2) then remove noises using algorithm 2
    3) finally compute the hedge density
    """
    start = time.time()
    print("step n13040303 - generate holes")
    hedge_mask = generate_transparent_holes(hedge_img, hedge_density)
    print("time = {}, step 2 - denoising".format(int(time.time() - start)))
    hedge_mask = hedge_denoising(hedge_mask)
    print("time = {}, step 3 - density calculation".format(int(time.time() - start)))
    hedge_density = compute_density(hedge_mask)
    hedge_density = round(hedge_density, 1)
    return hedge_mask, hedge_density


def generate_all_hedge_masks(resource_dir, save_dir):
    processed_count = 0
    for i, file in enumerate(os.listdir(resource_dir)):
        print("processing image {}: {}".format(i, file))
        hedge_path = os.path.join(resource_dir, file)
        try:
            hedge_img = Image.open(hedge_path)
            # generate hedge masks for every 5%
            for density in np.arange(0.1, 0.81, 0.05):
                print("processing density: {}".format(density))
                img = hedge_img.copy()
                hedge_mask, result_density = generate_hedge_mask(img, density)
                file_name = str(i + processed_count) + "_" + str(int(density * 100)) + ".png"
                save_path = os.path.join(save_dir, str(result_density), file_name)
                hedge_mask.save(save_path)

            dest = os.path.join("not used", "Completed_Hedges", file)
            os.replace(hedge_path, dest)
        except(OSError, NameError):
            print("OSError, Path:", hedge_path)


def generate_hedged_data_moving_background(background, hedge, crop_size):
    """
    First find the maximum overlapping area between hedge and bg
    Then crop a region with that size randomly and paste the mask on the bg
        - all the randomness added in is to increase the available training sample
    Then decide a random size of sliding window given a percentage

    Suggested frame number: 32
        for each frame, the window would be shifted randomly

    TBD: rotate image?
    """
    frame_num = 1
    hedge_width, hedge_height = hedge.size
    background_width, background_height = background.size

    # find the biggest overlap width and height possible

    if hedge_width < background_width:
        paste_width = hedge_width
        hedge_left = 0
        bg_left = np.random.randint(0, background_width - paste_width - 1)
    else:
        paste_width = background_width
        hedge_left = np.random.randint(0, hedge_width - paste_width - 1)
        bg_left = 0

    if hedge_height < background_height:
        paste_height = hedge_height
        hedge_top = 0
        bg_top = np.random.randint(0, background_height - paste_height - 1)
    else:
        paste_height = background_height
        hedge_top = np.random.randint(0, hedge_height - paste_height - 1)
        bg_top = 0

    # crop performed
    hedge_crop = hedge.crop((hedge_left, hedge_top, hedge_left + paste_width, hedge_top + paste_height))
    background_crop = background.crop((bg_left, bg_top, bg_left + paste_width, bg_top + paste_height))

    hedge_crop = hedge_crop.convert("RGBA")  # mask in paste() require input to be in RGBA
    crop_patches = [background_crop, hedge_crop]

    # rotate or mirror
    for idx in range(len(crop_patches)):
        if random_bool():
            crop_patches[idx] = ImageOps.flip(crop_patches[idx])
        if random_bool():
            crop_patches[idx] = ImageOps.mirror(crop_patches[idx])

    # masked background
    crop_patches[0].paste(crop_patches[1], (0, 0), mask=crop_patches[1])
    background_pasted = crop_patches[0]

    # sliding window related
    window_size = 0.4 if random_bool() else 0.7
    # in this way the sliding window will always have smaller size
    window_height = window_width = int(min(paste_height, paste_width) * window_size)

    # decide the initial top left corner for cropping
    paste_top = np.random.randint(0, paste_height - window_height - 1)
    paste_left = np.random.randint(0, paste_width - window_width - 1)

    # keep track of the bg location, the shifting will be operating on both coordinates
    bg_top += paste_top
    bg_left += paste_left

    frames = []
    for _ in range(frame_num):
        # for each frame, we shift the image randomly
        vertical_shift = np.random.randint(-4, 4)
        horizontal_shift = np.random.randint(-4, 4)

        # we have to check if the shift would cause the image to go out of bound
        while not (
                (paste_left + horizontal_shift) < 0 or (
                paste_left + window_width + horizontal_shift) > paste_width):
            horizontal_shift = np.random.randint(-8, 8)

        while not ((paste_top + vertical_shift) < 0 or (paste_top + window_height + vertical_shift) > paste_height):
            vertical_shift = np.random.randint(-8, 8)

        paste_top += vertical_shift
        paste_left += horizontal_shift

        # crop hedged frame
        hedged_frame = background_pasted.crop(
            (paste_left, paste_top, paste_left + window_width, paste_top + window_height))

        bg_top += vertical_shift
        bg_left += horizontal_shift

        # crop ground truth
        background_frame = background.crop(bg_left, bg_top, bg_left + window_width, bg_top + window_height)

        # get the hedged image and the non-hedged image
        frames.append((hedged_frame, background_frame))
    return frames


# def transfer_completed_hedges():
# #     try:
# #         path = os.path.join('Completed_Hedges')
# #         os.mkdir(path)
# #     except OSError:
# #         print("Directory exists")
# #
# #     file = open('data.txt')
# #     completed = file.readline()
# #     completed = completed[1::]
# #     completed = completed[:-1]
# #     completed = completed.split(',')
# #     for item in completed:
# #         item = item.replace("\"", "")
# #         item = item.replace(" ", "")
# #         source = os.path.join('Hedges', item)
# #         dest = os.path.join('Completed_Hedges', item)
# #         os.replace(source, dest)


# resource_directory = os.path.join('not used', 'Hedges')
# save_directory = 'Hedge_masks'
# generate_all_hedge_masks(resource_directory, save_directory)

def generate_all_hedged_images(hedge_path, background_path, target_path, logger, sample_no_per_image=10):
    # we generate sample_no_per_image amount of hedged data for each background image

    # the directory of the target path would be for example:
    # hedged data        /synthetic_data/test/1/ILSVRC2012_val_00009/0.1/001.jpeg
    # hedged data        /synthetic_data/test/1/ILSVRC2012_val_00009/0.1/002.jpeg
    # hedged data        /synthetic_data/test/1/ILSVRC2012_val_00009/0.1/003.jpeg
    # ground truth       /synthetic_data/test/1/ILSVRC2012_val_00009/ground_truth.jpeg

    # first we iterate through the background data
    for vgg_class in os.listdir(background_path):
        logger.info("start processing vgg class {}".format(vgg_class))
        dest_dir_path = os.path.join(target_path, vgg_class)
        source_dir_path = os.path.join(background_path, vgg_class)

        # for each class we create a new directory (../1)
        make_dir(dest_dir_path)

        # then we iterate through the background images
        for image_name in os.listdir(source_dir_path):
            logger.info("start processing image {} at vgg class {}".format(image_name, vgg_class))
            dest_background_path = os.path.join(dest_dir_path, os.path.splitext(image_name)[0])
            source_background_path = os.path.join(source_dir_path, image_name)

            # for each background images in class we create a new directory (../1/ILSVRC2012_val_00009)
            make_dir(dest_background_path)

            background = Image.open(source_background_path)

            # then we iterate through the hedge with different densities
            for hedge_density in os.listdir(hedge_path):
                dest_density_path = os.path.join(dest_background_path, hedge_density)
                source_density_path = os.path.join(hedge_path, hedge_density)
                hedge_masks = os.listdir(source_density_path)

                # for each density create a new directory for each image (../1/ILSVRC2012_val_00009/0.1)
                make_dir(dest_density_path)
                # save the ground truth
                ground_truth = background.copy()
                ground_truth = ground_truth.resize((128, 128))
                ground_truth.save(os.path.join(dest_background_path, image_name))

                # for each density generate hedged data we generate sample_no_per_image amount of hedged data
                for i in range(sample_no_per_image):
                    current_background = background.copy()
                    random_mask = Image.open(os.path.join(source_density_path, np.random.choice(hedge_masks)))
                    masked_background, hedge_mask = generate_single_hedged_image(current_background, random_mask)
                    # save them in the directory (../1/ILSVRC2012_val_00009/0.1/1.jpeg)
                    masked_background.save(os.path.join(dest_density_path, str(i) + ".jpeg"))
                    hedge_mask.save(os.path.join(dest_density_path, str(i) + "_hedge.png"))


def generate_single_hedged_image(input_img, hedge_mask):
    crop_width = crop_height = 1000
    current_background = input_img.copy()

    hedge_width, hedge_height = hedge_mask.size
    if min(hedge_width, hedge_height) < max(crop_width, crop_height):
        crop_width = crop_height = min(hedge_width, hedge_height)

    current_background = current_background.resize((crop_width, crop_height))

    # find a random point to crop
    left = np.random.randint(0, hedge_width - crop_width + 1)
    top = np.random.randint(0, hedge_height - crop_height + 1)

    current_hedge_mask = hedge_mask.crop((left, top, left + crop_width, top + crop_height))
    current_background.paste(current_hedge_mask, (0, 0), mask=current_hedge_mask)
    current_background = current_background.resize((128, 128))
    current_hedge_mask = current_hedge_mask.resize((128, 128))

    return current_background, current_hedge_mask


# because the ground truth was stored as its original size
def repair_ground_truth(root_dir):
    """ used to correct the size for stored ground truth image files """
    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        for image in os.listdir(label_path):
            image_path = os.path.join(label_path, image, image + ".JPEG")
            ground_truth_image = Image.open(image_path)
            ground_truth_image = ground_truth_image.resize((128, 128))
            ground_truth_image.save(image_path)

def resize_real_data(root_dir):
    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        for image in os.listdir(label_path):
            image_path = os.path.join(label_path, image)
            ground_truth_image = Image.open(image_path)
            ground_truth_image = ground_truth_image.resize((128, 128))
            ground_truth_image.save(image_path)