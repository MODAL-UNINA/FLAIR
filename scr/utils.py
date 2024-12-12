import os
import orjson
from tqdm import tqdm
import random
import copy
from collections import defaultdict
from itertools import repeat
import collections.abc
from torchvision import transforms
from PIL import Image
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# text complate
CUSTOM_TEMPLATES = {
    'label_with_description': 'a photo of a {}, {}.',
    'only_label': 'a photo of a {}.'
}

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
# ])


def image_transform(resize_size, is_train=True, augment_methods=None):
    """
    Create an image transformation function that supports multiple augmentation methods.

    Args:
        resize_size (int): Resize the image.
        is_train (bool): Whether it is training mode, determines whether to enable augmentation.
        augment_methods (list): A list of additional augmentation methods, each method will be used separately.

    Returns:
        list: A list of transforms.Compose objects for each augmentation method.
    """
    base_transform = [
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
    ]
    
    if is_train:
        transforms_list = []
        # Include base transform as one of the augmentations
        transforms_list.append(transforms.Compose(base_transform))
        
        # Add each augmentation method
        if augment_methods:
            transforms_list.extend([
                transforms.Compose([method] + base_transform) for method in augment_methods
            ])
        return transforms_list
    else:
        # Test set only needs the base transform
        return [transforms.Compose(base_transform)]



def get_data(client_data, transform_funcs):
    """
    For each sample, apply multiple augmentation methods.

    Args:
    client_data (list): Each sample contains (image_path, text).
    transform_funcs (list): Contains various transforms.Compose objects.

    Returns:
    list: The augmented data, containing multiple augmented versions of the same image.
    """
    augmented_data = []
    for impath, text in tqdm(client_data, desc="Processing augmented data"):
        img = Image.open(impath).convert("RGB")
        
        # Apply each transform
        for transform_func in transform_funcs:
            transformed_img = transform_func(img)
            augmented_data.append((transformed_img, text))
    
    return augmented_data


def split_client(json_data, image_dir, num_splits, seed, label_tag, description):

    # Step 1: Group the dataset by source
    grouped_data = defaultdict(list)
    for item in json_data:
        source = item["source"]
        grouped_data[source].append(item)
    
    # Step 2: Flatten grouped data while maintaining source-grouped integrity
    grouped_items = list(grouped_data.values())
    
    # Step 3: Shuffle the grouped items list for random assignment
    random.seed(seed)
    random.shuffle(grouped_items)
    
    # Step 4: Allocate each group to a client split
    splits = [[] for _ in range(num_splits)]
    for i, group in enumerate(grouped_items):
        splits[i % num_splits].extend(group)
    
    # Step 5: Process each split into samples
    total_clients = {}
    for i, split_dataset in enumerate(splits):
        samples = []
        for item in tqdm(split_dataset, desc=f"Processing client {i}"):
            impath = os.path.join(image_dir, item["image"])
            label = item.get(label_tag, "")
            description_text = item.get('description', "")

            # Tokenize text
            template = CUSTOM_TEMPLATES['label_with_description'] if description else CUSTOM_TEMPLATES['only_label']
            text = template.format(label, description_text) if description else template.format(label)
            text = clip.tokenize(text)

            samples.append((impath, text))
        total_clients[str(i)] = samples
    
    return total_clients


def split_train_test(total_clients, test_ratio, seed):
    train_data = {}
    test_data = []
    rng = random.Random(seed)
    
    for client_id, client_data in total_clients.items():
        train_size = int(len(client_data) * (1 - test_ratio))
        
        rng.shuffle(client_data)
        
        train_data[client_id] = client_data[:train_size]
        test_data.extend(client_data[train_size:])
    
    return train_data, test_data


def read_from_json(dataset_json):
    with open(dataset_json, "rb") as f:
        dataset = orjson.loads(f.read())
        
        num_items = len(dataset)
        print("Dataset size:", num_items)

    return dataset


def fedavg(w, weights):
    w_global = copy.deepcopy(w[0])
    num_clients = len(w)
    for k in w_global.keys():
        w_global[k] = w[0][k] * weights[0]
        for i in range(1, num_clients):
            w_global[k] += w[i][k] * weights[i]

    return w_global


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)

