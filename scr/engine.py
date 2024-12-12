import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

def count_trainable_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return trainable_params, total_params


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, text = self.dataset[item]

        return image, text


class LocalUpdate(object):
    def __init__(self, args, client_data):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        dataset = CustomDataset(client_data)
        self.train_loader = DataLoader(dataset, batch_size=self.args.train_bs, shuffle=True)

    def train(self, net):

        if self.args.PEFT=='adapter_image':
                for name, param in net.named_parameters():
                    if 'adapter_image' not in name:
                        param.requires_grad_(False)

        elif self.args.PEFT=='adapter_text':
                for name, param in net.named_parameters():
                    if 'adapter_text' not in name:
                        param.requires_grad_(False)

        elif self.args.PEFT=='adapter_two':
                for name, param in net.named_parameters():
                    if 'adapter_image' not in name and 'adapter_text' not in name:
                        param.requires_grad_(False)

        elif self.args.PEFT is None:
                for name, param in net.named_parameters():
                        param.requires_grad_(True)
        
        else:
            raise ValueError("Invalid PEFT setting")
    
        # Setting up the optimizer and learning rate scheduler
        params_to_update = [param for param in net.parameters() if param.requires_grad]
        optimizer = AdamW(params_to_update, lr=self.args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.local_ep, eta_min=self.args.eta_min)  
        # The learning rate is dynamically adjusted in the form of a cosine function. 

        # Mixed Precision scaler and Gradient Clipping
        scaler = GradScaler()
        max_norm = 1.0  # Set the maximum norm of gradient clipping

        net.train()

        trainable_params, total_params = count_trainable_params(net)
        print(f"Trainable Parameters: {trainable_params}")
        print(f"Total Parameters: {total_params}")

        # print("Trainable parameters:")
        # for name, param in net.named_parameters():
        #     if param.requires_grad:
        #         print(f"  {name}")

        epoch_loss = []

        for epoch in range(self.args.local_ep):
            batch_loss = []
            for batch in self.train_loader:
                images = batch[0].to(self.args.device)
                input_ids = batch[1].to(self.args.device)

                optimizer.zero_grad()

                # Mixed Precision training
                with autocast():
                    image_features, text_features, logit_scale_exp = net(images, input_ids)
                    # image_features, text_features, logit_scale_exp = quantized_model(images, input_ids)
                    logits_per_image = logit_scale_exp * (image_features @ text_features.T)
                    # logits_per_text = logit_scale_exp * (text_features @ image_features.T)

                    labels = torch.arange(len(images)).to(self.args.device)
                    # Contrastive loss
                    # loss = (self.loss_func(logits_per_image, labels) + self.loss_func(logits_per_text, labels)) / 2
                    loss = self.loss_func(logits_per_image, labels)

                scaler.scale(loss).backward()   # Gradient scaling and back propagation calculation
                scaler.unscale_(optimizer)      # Gradient unscale
                torch.nn.utils.clip_grad_norm_(params_to_update, max_norm)  # Gradient Clipping
                scaler.step(optimizer)
                scaler.update()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            # Update the learning rate scheduler 
            scheduler.step()

        if self.args.PEFT=='adapter_image':
            return net.adapter_image.state_dict(), sum(epoch_loss) / len(epoch_loss)
        
        elif self.args.PEFT=='adapter_text':
            return net.adapter_text.state_dict(), sum(epoch_loss) / len(epoch_loss)
        
        elif self.args.PEFT=='adapter_two':
            return net.adapter_image.state_dict(), net.adapter_text.state_dict(), sum(epoch_loss) / len(epoch_loss)

        elif self.args.PEFT==None:
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

        else:
            ValueError


def test(net, test_data, args):
    test_dataset = CustomDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False, drop_last=True)
    
    net.eval()

    correct_1 = 0
    correct_3 = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(args.device)
            input_ids = batch[1].to(args.device)
            labels = torch.arange(len(images)).to(args.device)

            image_features, text_features, logit_scale_exp = net(images, input_ids)

            logits_per_image = logit_scale_exp * (image_features @ text_features.T)

            # Top-1
            image_to_text_pred_1 = logits_per_image.topk(k=1, dim=1).indices
            correct_image_to_text_1 = (image_to_text_pred_1 == labels.unsqueeze(1)).any(dim=1).sum().item()
            correct_1 += correct_image_to_text_1

            # Top-3
            image_to_text_pred_3 = logits_per_image.topk(k=3, dim=1).indices
            correct_image_to_text_3 = (image_to_text_pred_3 == labels.unsqueeze(1)).any(dim=1).sum().item()
            correct_3 += correct_image_to_text_3

            total_samples += images.size(0)

    top_1 = correct_1 / total_samples * 100
    top_3 = correct_3 / total_samples * 100

    return top_1, top_3
