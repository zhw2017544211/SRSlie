os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os
import numpy as np
import torch
import argparse
import random
import shutil
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
from net.ft import ft
from net.eh import eh
from data import get_training_set, get_eval_set
import torchvision.models as models
from torchvision import transforms
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator



parser = argparse.ArgumentParser(description='SRSlie')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--seed', type=int, default=123, help='random seed to use.')
parser.add_argument('--data_train', type=str, default='dataset/train/')
parser.add_argument('--save_folder', default='checkpoint/')
parser.add_argument('--output_folder', default='results/')
parser.add_argument('--ftm', default='checkpoint/ftm.pth')
opt = parser.parse_args()


def seed_torch(seed=opt.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_torch()
cudnn.benchmark = True


def estimate_retinex(input_image, eps=1e-6):
    L, _ = torch.max(input_image, dim=1, keepdim=True)  # (B, 1, H, W)
    L = F.avg_pool2d(L, kernel_size=3, stride=1, padding=1)
    R = input_image / (L + eps)
    R = torch.clamp(R, 0.0, 1.0)
    return L, R


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


class Loss1(nn.Module):
    def __init__(self, lambda1=100, lambda2=1, lambda3=10):
        super().__init__()
        self.C_loss = nn.MSELoss()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3


    def forward(self, RR1, RR2, R1, p1, LL1, smoothed_L):

        loss1 = self.C_loss(RR1, RR2) * self.lambda1
        denominator = LL1.detach() + 1e-4
        target_R1 = p1 / denominator
        loss2 = self.C_loss(R1, target_R1) * self.lambda2
        loss3 = self.C_loss(LL1, smoothed_L) * self.lambda3

        total_loss = (loss2 + loss3 + loss4)
        return total_loss, {'loss1': loss1.item(), 'loss2': loss2.item(), 'loss3': loss3.item()}



class Loss2(torch.nn.Module):
    def __init__(self, E=0.6, lambda_exp=10, lambda_tv=1, lambda_color=10, lambda_sem=50):
        super().__init__()
        self.E = E
        self.lambda_exp = lambda_exp
        self.lambda_tv = lambda_tv
        self.lambda_color = lambda_color
        self.lambda_sem = lambda_sem

    def exposure_loss(self, L, patch_size=16):
        pool = torch.nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)
        L_patch = pool(L)
        loss = ((L_patch - self.E) ** 2).mean()
        return loss

    def total_variation_loss(self, L):
        loss = torch.mean(torch.abs(L[:, :, :-1, :] - L[:, :, 1:, :])) + \
               torch.mean(torch.abs(L[:, :, :, :-1] - L[:, :, :, 1:]))
        return loss

    def color_constancy_loss(self, I_enh):
        mean_rgb = torch.mean(I_enh, dim=(2, 3))  # (B, 3)
        R_mean, G_mean, B_mean = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        loss = ((R_mean - G_mean) ** 2 + (R_mean - B_mean) ** 2 + (G_mean - B_mean) ** 2).mean()
        return loss

    def semantic_consistency_loss(self, R, semantic_masks):
        B, C, H, W = R.shape
        total_loss = 0.0
        for b in range(B):
            mask_b = semantic_masks[b]
            R_b = R[b]
            num_regions = mask_b.shape[0]
            for k in range(num_regions):
                Mk = mask_b[k]
                Mk = Mk.unsqueeze(0)
                num_pixels = Mk.sum() + 1e-6
                mean_Rk = (R_b * Mk).sum(dim=(1, 2)) / num_pixels  # (3,)
                diff = (R_b - mean_Rk.view(3, 1, 1)) * Mk
                var = (diff ** 2).sum() / num_pixels
                total_loss += var
        total_loss = total_loss / B
        return total_loss

    def forward(self, L, R, I_enh, semantic_masks):
        L_exp = self.exposure_loss(L)
        L_tv = self.total_variation_loss(L)
        L_color = self.color_constancy_loss(I_enh)
        L_sem = self.semantic_consistency_loss(R, semantic_masks)

        total_loss = (self.lambda_exp * L_exp +
                      self.lambda_tv * L_tv +
                      self.lambda_color * L_color +
                      self.lambda_sem * L_sem)
        return total_loss, {'exp': L_exp.item(), 'tv': L_tv.item(), 'color': L_color.item(), 'sem': L_sem.item()}



def train():
    torch.set_grad_enabled(False)
    model.eval()
    ftm.eval()
    loss_print = 0

    for iteration, batch in enumerate(training_data_loader, 1):
        with torch.no_grad():
            X1, X2, file1, file2 = batch[0], batch[1], batch[2], batch[3]
        X1 = X1.cuda()
        X2 = X2.cuda()
        L1, R1 = estimate_retinex(X1)
        L2, R2 = estimate_retinex(X2)

        # stage1
        # L1, R1, p1 = ftm(X1, L1, R1)
        # L2, R2, p2 = ftm(X2, L2, R2)

        # sam
        I_toseg = torch.pow(L1, 0.2) * R1  #预增强
        I_toseg_np = I_toseg.detach().cpu().numpy()
        I_toseg_np = I_toseg_np[0].transpose(1, 2, 0)
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(I_toseg_np)
        b, c, h, w = I_toseg.shape
        transparent_image = np.zeros((h, w, 4), dtype=np.uint8)
        for ann in masks:
            mask = ann['segmentation']
            color_mask = np.random.random((1, 3)).tolist()[0]  
            transparent_image[mask > 0, 0] = color_mask[0] * 255
            transparent_image[mask > 0, 1] = color_mask[1] * 255
            transparent_image[mask > 0, 2] = color_mask[2] * 255
            transparent_image[mask > 0, 3] = 255
        Iseg = transparent_image.astype(np.float32) / 255.0
        Iseg_rgb = Iseg[:, :, :3]
        Iseg_rgb = Iseg_rgb.transpose(2, 0, 1)

        I_seg = Iseg_rgb[np.newaxis, :, :, :]
        I_seg_tensor = torch.from_numpy(I_seg).float()
        I_seg_tensor = I_seg_tensor.cuda()
    torch.set_grad_enabled(True)
    ehm.train()
    LL1, RR1, im1e = ehm(X1, L1, R1, I_seg_tensor)

    #loss
    enhanced = im1e
    # loss_fn = Loss1()
    loss_fn = Loss2()
    loss, loss_dict = loss_fn(LL1, RR1, enhanced, I_seg_tensor)

    print(f'loss{loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_print = loss_print + loss.item()
    torch.nn.utils.clip_grad_norm_(ehm.parameters(), max_norm=1.0)

    if iteration % 10 == 0:
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                                                                                    iteration,
                                                                                    len(training_data_loader),
                                                                                    loss_print,
                                                                                    optimizer.param_groups[0][
                                                                                        'lr']))
        loss_print = 0


def checkpoint(epoch):
    model_out_path = opt.save_folder + "epoch_{}.pth".format(epoch)
    torch.save(ehm.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU")
train_set = get_training_set(opt.data_train)
training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batchSize, shuffle=True)
ftm = ft().cuda()
ftm.load_state_dict(torch.load(opt.ftm, map_location=lambda storage, loc: storage))
ehm = eh().cuda()
optimizer = optim.Adam(ehm.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
milestones = []
for i in range(1, opt.nEpochs + 1):
    if i % 100 == 0:
        milestones.append(i)
scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)

score_best = 0

sam_checkpoint = 'checkpoint/sam_vit_h_4b8939.pth'
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

for epoch in range(1, opt.nEpochs + 1):
    train()
    scheduler.step()
    if epoch % 10 == 0:
        checkpoint(epoch)
