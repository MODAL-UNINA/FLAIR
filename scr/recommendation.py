import os
import numpy as np
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
import argparse
import clip
from utils import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="../ILID/data/", help="path to dataset")
parser.add_argument('--no_clients', default=5, help="number of clients: K")
parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
parser.add_argument('--seed', type=int, default=0, help="random seed (default: 1)")

args = parser.parse_args("")
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

#%%
def get_sample(client_data, transform_func):
    sample_data = []
    for impath, text in tqdm(client_data, desc="Processing augmented data"):
        img = Image.open(impath).convert("RGB")
        
        # Apply the transformation
        transformed_img = transform_func(img)
        sample_data.append((transformed_img, text))
    
    return sample_data

def get_feature(model, client_data):
    # Convert client data into a dataset and loader
    client_dataset = CustomDataset(client_data)
    client_loader = DataLoader(client_dataset, batch_size=32, shuffle=False, drop_last=False)

    # Initialize lists to store features
    img_feats = []
    txt_feats = []

    with torch.no_grad():
        for batch in client_loader:
            # Load data onto the specified device
            images = batch[0].to(args.device)
            input_ids = batch[1].to(args.device)
            
            # Forward pass to extract features
            image_features, text_features, _ = model(images, input_ids)

            # Accumulate features
            img_feats.append(image_features)
            txt_feats.append(text_features)

    # Concatenate all features into single tensors
    img_feats = torch.cat(img_feats, dim=0)
    txt_feats = torch.cat(txt_feats, dim=0)

    return img_feats, txt_feats

def smilarity(client_space, text_features):
    top_matches = []

    for client_idx, (img_feats, _) in client_space.items():
        # Calculate the cosine similarity between image and text feature
        cos_sim = F.cosine_similarity(img_feats, text_features, dim=-1)
        
        # get the idx
        max_similarity, img_idx = cos_sim.max(dim=0)
        
        # Record client index, highest similarity, image index
        top_matches.append((client_idx, max_similarity.item(), img_idx.item()))

    # Sort by similarity and select the first three
    top_3_matches = sorted(top_matches, key=lambda x: x[1], reverse=True)[:3]

    return top_3_matches

#%%
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

transform_train=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
    ])

# Apply the function to each client data
train_data = {
    client_id: get_sample(client_data, transform_func=transform_train)
    for client_id, client_data in train_data.items()
}

# model
model_path = "/home/modal-workbench/Projects/Pian/IIoT/ILID/main_data_augmetation_new/results/original_adapter_two_label_long/model.pt"
model = torch.load(model_path, map_location=torch.device(args.device))
model.eval()

# for 5 clients, get 5 image feature space
client_space = {
    idx: get_feature(model, client_data)
    for idx, client_data in train_data.items()
}

#%%
# now revived a NLP qury
# query = 'u-handle'
# query = 'floor cleaning'
query = 'oil sight glasses'
query = str(query)
# Tokenize text
TEMPLATES = {'query': 'a photo of a {}.'}

template = TEMPLATES['query']
text = template.format(query)
text = clip.tokenize(text)
text = text.to(args.device)

text_features = model.encode_text(text)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
logit_scale_exp = model.logit_scale.exp()

# calculate smilarity, get top 3 mathes
top_3_matches = smilarity(client_space, text_features)

top_images = []
top_titles = []
for rank, (client_idx, similarity, img_idx) in enumerate(top_3_matches, start=1):
    image = train_data[client_idx][img_idx][0]
    description = train_data[client_idx][img_idx][1]
    print(f"Rank {rank}: Client {client_idx}, Image {img_idx} with similarity {similarity:.3f}")
    top_images.append(image)
    top_titles.append(f"From Client {client_idx}\nSimilarity: {similarity:.3f}")

num_images = len(top_images)

fig, axes = plt.subplots(1, min(3, num_images), figsize=(10, 3))

for i, image_tensor in enumerate(top_images):
    image_tensor = image_tensor.cpu()

    # Denormalization
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.25, 0.25, 0.25]).view(3, 1, 1)
    image_tensor = image_tensor * std + mean
    image_tensor = image_tensor.clamp(0, 1)
    image_np = image_tensor.permute(1, 2, 0).numpy()

    axes[i].imshow(image_np)
    axes[i].axis('off')

    axes[i].text(0, 1, top_titles[i], fontsize=16, ha='left', va='top', transform=axes[i].transAxes)

    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="black", alpha=0.7)
    axes[i].text(0, 1, top_titles[i], fontsize=16, ha='left', va='top', color='white', 
                 transform=axes[i].transAxes, bbox=bbox_props)

plt.tight_layout()
plt.show()

