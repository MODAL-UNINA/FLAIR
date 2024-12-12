import os
import json
import numpy as np
import random
import torch
from torchvision import transforms
import time
import copy
import argparse
import psutil
import clip
from utils import *
from model import *
from engine import *
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="../FLARE/data/", help="path to dataset")
parser.add_argument("--model_root", type=str, default="../FLARE/scr/model/", help="path to model")
parser.add_argument("--save_dir", type=str, default="../FLARE/results", help="path to model")
parser.add_argument("--model", type=str, default="vit_b_16", help="vit_b_16, vit_t_16(Distilled)")
parser.add_argument('--no_clients', default=5, help="number of clients: K")
parser.add_argument('--rounds', type=int, default=50, help="rounds of communication")
parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
parser.add_argument('--train_bs', type=int, default=64, help="batch size of training")
parser.add_argument('--test_bs', type=int, default=32, help="batch size of test")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--eta_min', type=float, default=1e-5, help="eta_min of CosineAnnealingLR")
parser.add_argument("--label_tag", type=str, default="label_long", help="use which label: label_short, label_long")
parser.add_argument("--description", type=bool, default=False, help="use description or not")
parser.add_argument('--reduction', type=int, default=4, help="")
parser.add_argument("--PEFT", type=str, default="adapter_two", help="use which PEFT: None, adapter_image, adapter_text, adapter_two")
parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
parser.add_argument('--seed', type=int, default=0, help="random seed (default: 1)")

args = parser.parse_args("")
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

image_dir = os.path.join(args.data_root, "images")
json_data = read_from_json(os.path.join(args.data_root, "ilid.json"))

total_clients = split_client(
    json_data=json_data,
    image_dir=image_dir,
    num_splits=args.no_clients,
    seed=args.seed,
    label_tag=args.label_tag,
    description=args.description
)

train_data, test_data = split_train_test(total_clients, test_ratio=0.2, seed=args.seed)

# 4 different augmentation method
augment_methods = [
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(214, scale=(0.8, 1.0)),
]

# transform
transform_train = image_transform(resize_size=224, is_train=True, augment_methods=augment_methods)
transform_test = image_transform(resize_size=224, is_train=False)

train_data = {
    client_id: get_data(client_data, transform_train)
    for client_id, client_data in train_data.items()
}

test_data = get_data(test_data, transform_test)


if args.model == "vit_t_16":
    model_cfg_file = 'ViT-T-16.json'
    model_state_dict = torch.load(os.path.join(args.model_root, "ViT-B-16_cc3m_12m_kd_ViT-T-16_cc3m_12m_ep32.pt"))['state_dict']
    model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
    
elif args.model == "vit_b_16":
    model_cfg_file = 'ViT-B-16.json'
    original_model, preprocess = clip.load("ViT-B/16", device=args.device)
    model_state_dict = original_model.state_dict()
    for key in ["input_resolution", "context_length", "vocab_size"]:
        model_state_dict.pop(key, None)
    
else:
    raise ValueError("Invalid model type")

with open(os.path.join(os.getcwd(), os.path.join(args.model_root, model_cfg_file)), 'r') as f:
    model_cfg = json.load(f)


model = create_model(args, model_cfg)
model.load_state_dict(model_state_dict, strict=False)


# training
loss_train = []
top_1_list, top_3_list = [], []

# time
start_time = time.time()

for iter in range(args.rounds):

    loss_locals = []
    w_locals = []
    w_adapter_locals_image = []
    w_adapter_locals_text = []

    # maybe weighted
    client_sizes_list = [len(client_data) for client_id, client_data in train_data.items()]
    total_size = sum(client_sizes_list)
    weights = [size / total_size for size in client_sizes_list]

    for idx, client_data in train_data.items():

        if args.PEFT=='adapter_image':
            local = LocalUpdate(args, client_data=client_data)
            w_adapter_image, loss = local.train(net=copy.deepcopy(model).to(args.device))
            w_adapter_locals_image.append(copy.deepcopy(w_adapter_image))

        elif args.PEFT=='adapter_text':
            local = LocalUpdate(args, client_data=client_data)
            w_adapter_text, loss = local.train(net=copy.deepcopy(model).to(args.device))
            w_adapter_locals_text.append(copy.deepcopy(w_adapter_text))

        elif args.PEFT=='adapter_two':
            local = LocalUpdate(args, client_data=client_data)
            w_adapter_image, w_adapter_text, loss = local.train(net=copy.deepcopy(model).to(args.device))
            w_adapter_locals_image.append(copy.deepcopy(w_adapter_image))
            w_adapter_locals_text.append(copy.deepcopy(w_adapter_text))

        elif args.PEFT is None:
            local = LocalUpdate(args, client_data=client_data)
            w, loss = local.train(net=copy.deepcopy(model).to(args.device))
            w_locals.append(copy.deepcopy(w))

        else:
            ValueError

        loss_locals.append(loss)

    # update global weights by weight average
    if args.PEFT=='adapter_image':
        w_adapter_glob_image = fedavg(w_adapter_locals_image, weights)
        model.adapter_image.load_state_dict(w_adapter_glob_image)

    elif args.PEFT=='adapter_text':
        w_adapter_glob_text = fedavg(w_adapter_locals_text, weights)
        model.adapter_text.load_state_dict(w_adapter_glob_text)

    elif args.PEFT=='adapter_two':
        w_adapter_glob_image = fedavg(w_adapter_locals_image, weights)
        model.adapter_image.load_state_dict(w_adapter_glob_image)
        w_adapter_glob_text = fedavg(w_adapter_locals_text, weights)
        model.adapter_text.load_state_dict(w_adapter_glob_text)

    elif args.PEFT is None:
        w_glob = fedavg(w_locals, weights)
        model.load_state_dict(w_glob)

    # print train loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    print('Round {:3d}, Average Train loss {:.3f}'.format(iter, loss_avg))
    loss_train.append(loss_avg)

    top_1, top_3 = test(model, test_data, args)
    print('Test acc (Top-1): {:.3f}'.format(top_1))
    print('Test acc (Top-3): {:.3f}'.format(top_3))

    top_1_list.append(top_1)
    top_3_list.append(top_3)


end_time = time.time()
cost_time = int(end_time - start_time)

memory_used = psutil.Process().memory_info().rss
memory_cost = int(memory_used / 1024 / 1024)


# save model
save_folder = os.path.join(args.save_dir, f"{args.model}_{args.PEFT}_{args.label_tag}")

os.makedirs(save_folder, exist_ok=True)

save_path = os.path.join(save_folder, "model.pt")
torch.save(model, save_path)

# save train loss
train_loss_path = os.path.join(save_folder, "train_loss.pkl")
with open(train_loss_path, 'wb') as f:
    pickle.dump({'loss_train': loss_train}, f)

# save top-1
test_top1_path = os.path.join(save_folder, "test_top1.pkl")
with open(test_top1_path, 'wb') as f:
    pickle.dump({'top_1': top_1_list}, f)

# save top-3
test_top3_path = os.path.join(save_folder, f"test_top3_time_{cost_time}_memory_{memory_cost}.pkl")
with open(test_top3_path, 'wb') as f:
    pickle.dump({'top_3': top_3_list}, f)
