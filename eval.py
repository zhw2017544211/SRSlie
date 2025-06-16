os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os
import time
import argparse
from thop import profile
from net.ft import ft
from net.eh import eh
from data import get_eval_set
from torchvision import transforms
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


parser = argparse.ArgumentParser(description='SRSlie')
parser.add_argument('--data_test', type=str, default='dataset/test')
parser.add_argument('--ftm', default='checkpoint/ftm.pth')
parser.add_argument('--ehm', default='checkpoint/ehm.pth')
parser.add_argument('--output_folder', type=str, default='results/')
opt = parser.parse_args()


test_set = get_eval_set(opt.data_test)
testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)
ftm = ft().cuda()
ftm.load_state_dict(torch.load(opt.ftm, map_location=lambda storage, loc: storage))
ehm = eh().cuda()
ehm.load_state_dict(torch.load(opt.ehm, map_location=lambda storage, loc: storage))
sam_checkpoint = 'checkpoint/sam_vit_h_4b8939.pth'
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


def estimate_retinex(input_image, eps=1e-6):
    L, _ = torch.max(input_image, dim=1, keepdim=True)  # (B, 1, H, W)
    L = F.avg_pool2d(L, kernel_size=3, stride=1, padding=1)
    R = input_image / (L + eps)
    R = torch.clamp(R, 0.0, 1.0)
    return L, R

def eval():
    torch.set_grad_enabled(False)
    ftm.eval()
    ehm.eval()
    for batch in testing_data_loader:
        with torch.no_grad():
            input, name = batch[0], batch[1]
        input = input.cuda()
        print(name)
        with torch.no_grad():
            L, R = estimate_retinex(input)
            #sam
            I_toseg = torch.pow(L, 0.2) * R  # 预增强
            I_toseg_np = I_toseg.detach().cpu().numpy()
            I_toseg_np = I_toseg_np[0].transpose(1, 2, 0)
            mask_generator = SamAutomaticMaskGenerator(sam)
            masks = mask_generator.generate(I_toseg_np)
            b, c, h, w = I_toseg.shape
            transparent_image = np.zeros((h, w, 4), dtype=np.uint8)
            for ann in masks:
                mask = ann['segmentation']
                color_mask = np.random.random((1, 3)).tolist()[0]
                # 为标注区域填充颜色
                transparent_image[mask > 0, 0] = color_mask[0] * 255  # R
                transparent_image[mask > 0, 1] = color_mask[1] * 255  # G
                transparent_image[mask > 0, 2] = color_mask[2] * 255  # B
                transparent_image[mask > 0, 3] = 255  # 透明度
            Iseg = transparent_image.astype(np.float32) / 255.0
            Iseg_rgb = Iseg[:, :, :3]
            Iseg_rgb = Iseg_rgb.transpose(2, 0, 1)
            I_seg = Iseg_rgb[np.newaxis, :, :, :]
            I_seg_tensor = torch.from_numpy(I_seg).float()
            I_seg_tensor = I_seg_tensor.cuda()
            L1, R1, p = ftm(X,L,R)
            LL, RR, I = ehm(X, L1, R1, I_seg_tensor)
            I_e = LL * RR
            L = L.cpu()
            R = R.cpu()
            LL = LL.cpu()
            RR = RR.cpu()
            I_e = I_e.cpu()

            L_img = transforms.ToPILImage()(L.squeeze(0))
            R_img = transforms.ToPILImage()(R.squeeze(0))
            LL_img = transforms.ToPILImage()(LL.squeeze(0))
            RR_img = transforms.ToPILImage()(RR.squeeze(0))
            I_e_img = transforms.ToPILImage()(I_e.squeeze(0))

            L_img.save(opt.output_folder + '/L/' + name[0])
            R_img.save(opt.output_folder + '/R/' + name[0])
            LL_img.save(opt.output_folder + '/LL/' + name[0])
            RR_img.save(opt.output_folder + '/RR/' + name[0])
            I_e_img.save(opt.output_folder + '/I/' + name[0])
        torch.cuda.empty_cache()
        torch.set_grad_enabled(True)

eval()







