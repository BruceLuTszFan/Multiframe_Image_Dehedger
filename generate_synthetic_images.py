import os
from utils import get_logger
from hedge_mask_utils import generate_all_hedged_images

data_type = 'test'

logger = get_logger()
hedge_path = os.path.join('data', 'Hedge_masks')
test_background_path = os.path.join('data', 'imagenet_data', data_type)
test_target_path = os.path.join('data', 'synthetic_data', data_type)

generate_all_hedged_images(hedge_path, test_background_path, test_target_path, logger, sample_no_per_image=3)
