import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
from scipy.ndimage import gaussian_filter
import math
from statistics import mean
from PIL import Image

import os
import glob
import sys
from pathlib import Path
from tqdm import tqdm
import time

sys.path.append(str(Path(__file__).parents[1]))
from utils import tensor2img

sys.path.append(str(Path(__file__).parents[3]))
from basicsr.utils.options import parse
from importlib import import_module


placeholder_dp = "path to save noisy images for denoising task"

#Path to the datasetfolder for inference
pth_to_dataset_folder = "/datasets/"


def ssim_calculate(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
    img1 = np.array(img1, dtype=np.float32) / 255
    img2 = np.array(img2, dtype=np.float32) / 255
    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return np.mean(ssim_map)

def calc_PSNR(img1, img2):
    '''
    img1 and img2 have range [0, 255]
    '''
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)



class Denoising(torch.utils.data.Dataset):
    def __init__(self, data_path, video,
                 noise_std, 
                 dataset_name="Set8", 
                 sample=True):
        # sample: if True, new data is created (since noise is random).
        super().__init__()
        self.data_path = data_path
        self.dataset = dataset_name
        if self.dataset == "DAVIStest":
            ext = ".jpg"
        if self.dataset == "Set8":
            ext = ".png"
        
        self.files = sorted(glob.glob(os.path.join(data_path, video, f'*{ext}')))
        self.len = self.bound = len(self.files)
        self.current_frame = 0
        print(f"> # of Frames in {video} of {self.dataset}: {len(self.files)}")
        self.transform = transforms.Compose([transforms.ToTensor()])

        Img = Image.open(self.files[0])
        Img = np.array(Img)
        H, W, C = Img.shape

        os.makedirs(os.path.join(f"{placeholder_dp}{self.dataset}/{video}_{int((noise_std)*255)}"), exist_ok=True)
        self.noisy_folder = os.path.join(f"{placeholder_dp}{self.dataset}/{video}_{int((noise_std)*255)}")

        if sample:
            for i in range(self.len):
                Img = Image.open(self.files[i])
                Img = self.transform(Img)
                self.C, self.H, self.W = Img.shape
                std1 = noise_std
                noise = torch.empty_like(Img).normal_(mean=0, std=std1) #.cuda().half()
                Img = Img + noise
                
                np.save(os.path.join(self.noisy_folder, os.path.basename(self.files[i])[:-3]+"npy"), Img)
        
        self.noisy_files = sorted(glob.glob(os.path.join(self.noisy_folder, "*.npy")))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        Img = Image.open(self.files[index])
        noisy_Img = np.load(self.noisy_files[index])


        img_gt = np.array(Img)
        img_gt = self.transform(img_gt)
        img_in = torch.from_numpy(noisy_Img.copy())

        return (img_gt.type(torch.FloatTensor), 
                img_in.type(torch.FloatTensor))

class VideoLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, video):
        super().__init__()
        self.data_path = data_path
        self.in_files = sorted(glob.glob(os.path.join(data_path, video + "/*.*")))
        self.gt_files = sorted(glob.glob(os.path.join(data_path.replace("blur", "gt"), video + "/*.*")))
        self.len = len(self.in_files)
        print(f"> # of Frames in {video}: {len(self.in_files)}")
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.reverse = transforms.Compose([transforms.ToPILImage()])
        
        Img = Image.open(self.in_files[0])
        Img = np.array(Img)
        H, W, C = Img.shape
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_in = Image.open(self.in_files[index])
        img_in = np.array(img_in)
        
        img_gt = Image.open(self.gt_files[index])
        img_gt = np.array(img_gt)

        return (self.transform(np.array(img_gt)).type(torch.FloatTensor), 
                self.transform(np.array(img_in)).type(torch.FloatTensor))


def run_inference_patched(img_lq_prev,
                          img_lq_curr,
                          model, device,
                          tile,
                          tile_overlap,
                          dataset_name,
                          prev_patch_dict_k=None, 
                          prev_patch_dict_v=None,
                          img_multiple_of = 8,
                          scale=1,
                          model_type='t0'):
    
    height, width = img_lq_curr.shape[2], img_lq_curr.shape[3]
    
    H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
    padh = H-height if height%img_multiple_of!=0 else 0
    padw = W-width if width%img_multiple_of!=0 else 0

    img_lq_curr = torch.nn.functional.pad(img_lq_curr, (0, padw, 0, padh), 'reflect')
    img_lq_prev = torch.nn.functional.pad(img_lq_prev, (0, padw, 0, padh), 'reflect')
        
    # test the image tile by tile
    b, c, h, w = img_lq_curr.shape
    tile = min(tile, h, w)
    assert tile % 8 == 0, "tile size should be multiple of 8"
    tile_overlap = tile_overlap

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    E = torch.zeros(b, c, h, w).type_as(img_lq_curr)
    W = torch.zeros_like(E)

    patch_dict_k = {}
    patch_dict_v = {}
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:

            in_patch_curr = img_lq_curr[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            in_patch_prev = img_lq_prev[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            
            # prepare for SR following EAVSR.
            if model_type == "SR":
                in_patch_prev = torch.nn.functional.interpolate(in_patch_prev, 
                                                                    scale_factor=1/4,
                                                                    mode="bicubic")
                in_patch_curr = torch.nn.functional.interpolate(in_patch_curr, 
                                                                    scale_factor=1/4, 
                                                                    mode="bicubic")
    
            x = torch.concat((in_patch_prev.unsqueeze(0), 
                              in_patch_curr.unsqueeze(0)), dim=1)
            x = x.to(device)

            if prev_patch_dict_k is not None and prev_patch_dict_v is not None:
                old_k_cache = [x.to(device) if x is not None else None for x in prev_patch_dict_k[f"{h_idx}-{w_idx}"]]
                old_v_cache = [x.to(device) if x is not None else None for x in prev_patch_dict_v[f"{h_idx}-{w_idx}"]]
            else:
                old_k_cache = None
                old_v_cache = None

            out_patch, k_c, v_c = model(x.float(),
                                        old_k_cache,
                                        old_v_cache)
            patch_dict_k[f"{h_idx}-{w_idx}"] = [x.detach().cpu() if x is not None else None for x in k_c]
            patch_dict_v[f"{h_idx}-{w_idx}"] = [x.detach().cpu() if x is not None else None for x in v_c]
            out_patch = out_patch.detach().cpu()
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
            W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
    
    restored = E.div_(W)
    restored = torch.clamp(restored, 0, 1)
    return restored, patch_dict_k, patch_dict_v

def load_model(path, model):
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path)['params'])
    model = model.to(device)
    model.eval()
    print(f"> Loaded Model.")
    return model, device

def flatten(xss):
    return [x for xs in xss for x in xs]

def run_inference(video_name, test_loader, 
                  model, device,
                  model_name,
                  save_img, do_patched, 
                  image_out_path, tile, 
                  tile_overlap, 
                  dataset_name,
                  y_channel_PSNR=False, 
                  model_type='t0'):
    
    previous_frame = None
    previous_gt = None

    per_video_score = []
    per_video_ssim = []

    k_cache, v_cache = None, None
    for ix in range(len(test_loader.dataset)):
        current_gt = test_loader.dataset[ix][0]
        current_frame = test_loader.dataset[ix][1]

        if previous_frame is None and previous_gt is None:
            previous_frame = current_frame
            
        c, h, w = current_gt.shape
        if do_patched:
            # do inference in patches, and concatenate before computing PSNR/SSIM.
            x2, k_cache, v_cache = run_inference_patched(
                                        previous_frame.unsqueeze(0),
                                        current_frame.unsqueeze(0),
                                        model, device, tile=tile, 
                                        tile_overlap=tile_overlap,
                                        dataset_name=dataset_name, 
                                        prev_patch_dict_k=k_cache, 
                                        prev_patch_dict_v=v_cache,
                                        model_type=model_type)
        else:
            # prepare for SR following EAVSR.
            if model_type == "SR":
                previous_frame = torch.nn.functional.interpolate(previous_frame, 
                                                                    scale_factor=1/4,
                                                                    mode="bicubic")
                current_frame = torch.nn.functional.interpolate(current_frame, 
                                                                    scale_factor=1/4, 
                                                                    mode="bicubic")
            # do inference on whole frame if the memory can be fit.
            x = torch.concat((previous_frame.unsqueeze(0), 
                            current_frame.unsqueeze(0)), dim=0).to(device)
            x2, k_cache, v_cache = model(x.unsqueeze(0), k_cache, v_cache)

        x2 = x2.squeeze(0)
        x2 = x2[:, :h, :w]
        
        if y_channel_PSNR:
            gt_255 = (current_gt*255.0).permute(1, 2, 0).numpy().astype(np.uint8)
            gt_255_bgr = gt_255[:, :, ::-1]
            gt_y_chnl = bgr2ycbcr(gt_255_bgr)
            x2_255 = (x2*255.0).permute(1, 2, 0).numpy().astype(np.uint8)
            x2_255_bgr = x2_255[:, :, ::-1]
            x2_y_chnl = bgr2ycbcr(x2_255_bgr)

            psnr_score = calc_PSNR(x2_y_chnl, gt_y_chnl)
            ssim_score = ssim_calculate(x2_y_chnl, gt_y_chnl)
        else:
            psnr_score = calc_PSNR(tensor2img(x2, rgb2bgr=False), 
                                tensor2img(current_gt, rgb2bgr=False))
            ssim_score = ssim_calculate(tensor2img(x2, rgb2bgr=False), 
                                    tensor2img(current_gt, rgb2bgr=False))
        
        
        print(f"PSNR for Frame: {ix} -- {psnr_score}")
        # print(f"SSIM for frame {ix} -- {ssim_score}")

        per_video_score.append(psnr_score)
        per_video_ssim.append(ssim_score)

        if save_img:
            fig, axs = plt.subplots(1, 3, figsize=(10,10))
            
            axs[0].imshow(current_frame.permute(1, 2, 0).detach().cpu().numpy())
            axs[1].imshow(x2.permute(1, 2, 0).detach().cpu().numpy())
            axs[2].imshow(current_gt.permute(1, 2, 0).detach().cpu().numpy())
            
            axs[0].set_title('Input')
            axs[1].set_title(f'Pred {psnr_score:.2f}/{ssim_score:.2f}')
            axs[2].set_title(f'GT Frame {ix}')
            plt.tight_layout()

            # Ensure the directory for the model exists
            base_path = image_out_path
            base_path = os.path.join(base_path, model_name)
            os.makedirs(base_path, exist_ok=True)
            base_path = os.path.join(base_path, video_name)
            os.makedirs(base_path, exist_ok=True)
            file_name = f"Frame_{ix+1}.png"  # ix should be defined in your loop or context
            file_name_inp = os.path.join(base_path, f"Frame_{ix+1}_Input.png")  # ix should be defined in your loop or context
            file_name_pred = os.path.join(base_path, f"Frame_{ix+1}_Pred.png")  # ix should be defined in your loop or context
            file_name_gt = os.path.join(base_path, f"Frame_{ix+1}_GT.png")  # ix should be defined in your loop or context
            
            save_path = os.path.join(base_path, file_name)
            plt.savefig(save_path, bbox_inches='tight')
            cv2.imwrite(file_name_pred, cv2.cvtColor((x2.permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            cv2.imwrite(file_name_inp, cv2.cvtColor((current_frame.permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            cv2.imwrite(file_name_gt, cv2.cvtColor((current_gt.permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_BGR2RGB))

        previous_frame = current_frame
        previous_gt = current_gt

    print(f"PSNR for {video_name}: {mean(per_video_score)}")
    print(f"SSIM for {video_name} is {mean(per_video_ssim)}")
    return per_video_score, per_video_ssim

def create_video_model(opt, model_type="t0"):
    if model_type == "t0":
        module = import_module('basicsr.models.archs.turtle_arch')
        model = module.make_model(opt)
    elif model_type == "t1":
        module = import_module('basicsr.models.archs.turtle_t1_arch')
        model = module.make_model(opt)
    elif model_type == "SR":
        module = import_module('basicsr.models.archs.turtle_super_t1_arch')
        model = module.make_model(opt)
    else:
        print("Model type not defined")
        exit()
    return model

def main(model_path,
         model_name,
         dataset_name,
         task_name,
         config_file,
         tile,
         tile_overlap,
         save_image,
         model_type,
         do_pacthes,
         image_out_path,
         noise_sigma=50.0/255.0,
         sample=True,
         y_channel_PSNR=False):

    print(f"dataset_name: {dataset_name}")
    print(f"task_name: {task_name}")
    print(f"model_type: {model_type}")
    print(f"do_patches: {do_pacthes}")
    print(f"tile: {tile}")
    print(f"tile_overlap: {tile_overlap}")
    print(F"sample: {sample}")
    
    if dataset_name == "DAVIStest":
        data_dir = pth_to_dataset_folder + "/datasets/DAVIS_testdev/DAVIS/JPEGImages/480p/"
    elif dataset_name == "GoPro":
        data_dir = pth_to_dataset_folder + "/datasets/GoPro/test/blur/"
    elif dataset_name == "RSVD":
        data_dir = pth_to_dataset_folder + "/datasets/Desnowing/rsvd/test/blur/"
    elif dataset_name == "NightRain":
        data_dir = pth_to_dataset_folder + "/datasets/NightRain/test/blur/"
    elif dataset_name == "Set8":
        data_dir = pth_to_dataset_folder + "/datasets/Set8/"
    elif dataset_name == "VRDS":
        data_dir = pth_to_dataset_folder + "/datasets/VRDS/test/rainstreak/blur/"
    elif dataset_name == "BSD":
        data_dir = pth_to_dataset_folder + "/datasets/BSD_3ms24ms/prepd_data/test/blur/"
    elif dataset_name == "MVSR":
        data_dir = pth_to_dataset_folder + "/datasets/MVSR4x/test/blur/"
    else:
        print(f"Invalid Options.")
        exit(0)

    opt = parse(config_file, is_train=True)
    model = create_video_model(opt, model_type)

    model, device = load_model(model_path, model)
    videos = sorted(glob.glob(os.path.join(data_dir, '*')))

    total_score_psnr = []
    total_score_ssim = []

    for video in videos:
        video_name = video.split('/')[-1]
        if video_name:
            if task_name == "Denoising":
                data = Denoising(data_dir,
                                video_name,
                                noise_sigma,
                                dataset_name=dataset_name,
                                sample=sample)
            else:
                data = VideoLoader(data_dir,
                                video_name)
                
            test_loader = torch.utils.data.DataLoader(data,
                                                    batch_size=1, 
                                                    num_workers=1, 
                                                    shuffle=False)
            per_video_score, per_video_ssim = run_inference(video_name,
                                                            test_loader,
                                                            model,
                                                            device,
                                                            model_name,
                                                            save_img=save_image,
                                                            do_patched=do_pacthes,
                                                            image_out_path=image_out_path,
                                                            tile=tile,
                                                            tile_overlap=tile_overlap,
                                                            dataset_name=dataset_name,
                                                            y_channel_PSNR=y_channel_PSNR,
                                                            model_type=model_type)
            total_score_psnr.append(per_video_score)
            total_score_ssim.append(per_video_ssim)
        
    total_psnr = flatten(total_score_psnr)
    total_ssim = flatten(total_score_ssim)

    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Total PSNR: {mean(total_psnr)}")
    print(f"Total SSIM: {mean(total_ssim)}")
    return total_psnr, total_ssim



if __name__ == "__main__":
    st = time.time()

    # #----------------------------------------------------------------------------------------------------------
    #Desnowing
    # config = "options/Turtle_Desnow.yml"
    # model_path = "trained_models/Desnow.pth"
    # model_name = "Gaia_Desnow_simple_full"
    # print(model_name)
    # _, _ = main(model_path=model_path,
    #             model_name=model_name, 
    #             config_file=config,

    #             dataset_name="RSVD", #GoPro, SR, NightRain, DVD, Set8
    #             task_name="Desnowing", #Deblurring, SR, Deraining, Deblurring, Denoising

    #             model_type="t0",

    #             save_image=True,
    #             image_out_path="/outputs/",

    #             do_pacthes=True,
    #             tile=320,
    #             tile_overlap=256)

    # end = time.time()
    # print(f"Completed in {end-st}s")


    # ----------------------------------------------------------------------------------------------------------
    # SR, MVSR Dataset
    config = "options/Turtle_SR_MVSR.yml"
    model_path = "trained_models/SuperResolution.pth"
    model_name = "SR"
    print(model_name)
    _, _ = main(model_path=model_path,
                model_name=model_name, 
                config_file=config,

                dataset_name="MVSR", #GoPro, MVSR, NightRain, DVD, Set8
                task_name="SuperResolution", #Deblurring, SuperResolution, Deraining, Deblurring, Denoising

                model_type="SR", #t0,t1,SR

                save_image=True,
                image_out_path="/outputs/",

                do_pacthes=True,
                tile=256,
                tile_overlap=64)

    end = time.time()
    print(f"Completed in {end-st}s")

    # # ----------------------------------------------------------------------------------------------------------
    # #Deraining, night
    # config = "options/Turtle_Derain.yml"
    # model_path = "trained_models/NightRain.pth"
    # model_name = "Turtle_Derain_simple_320_128_30"
    # print(model_name)
    # _, _ = main(model_path=model_path,
    #             model_name=model_name, 
    #             config_file=config,

    #             dataset_name="NightRain", #GoPro, SR, NightRain, DVD, Set8
    #             task_name="Deraining", #Deblurring, SR, Deraining, Deblurring, Denoising

    #             model_type="t0",

    #             save_image=True,
    #             image_out_path="/outputs/",

    #             do_pacthes=True,
    #             tile=320,
    #             tile_overlap=128,
                
    #             y_channel_PSNR=True)
    # end = time.time()
    # print(f"Completed in {end-st}s")

    # ----------------------------------------------------------------------------------------------------------
    # Deraining, raindrop

    # config = "options/Turtle_Derain_VRDS.yml"
    # model_path = "trained_models/RainDrop.pth"
    # model_name = "Turtle_RainDrop_simple_320_128"
    # print(model_name)
    # _, _ = main(model_path=model_path,
    #             model_name=model_name, 
    #             config_file=config,

    #             dataset_name="VRDS", #GoPro, SR, NightRain, DVD, Set8
    #             task_name="Deraining", #Deblurring, SR, Deraining, Deblurring, Denoising

    #             model_type="t1",

    #             save_image=True,
    #             image_out_path="/outputs/",

    #             do_pacthes=True,
    #             tile=320,
    #             tile_overlap=128)

    # end = time.time()
    # print(f"Completed in {end-st}s")

    #----------------------------------------------------------------------------------------------------------
    # Deblurring, Gopro
    # config = "options/Turtle_Deblur_Gopro.yml"
    # model_path = "trained_models/GoPro_Deblur.pth"
    # model_name = "Turtle_GoPro_simple_320_128_200k_kamran_no_pos"
    # print(model_name)
    # _, _ = main(model_path=model_path,
    #             model_name=model_name, 
    #             config_file=config,

    #             dataset_name="GoPro", #GoPro, SR, NightRain, DVD, Set8
    #             task_name="Deblurring", #Deblurring, SR, Deraining, Deblurring, Denoising

    #             model_type="t1",

    #             save_image=True,
    #             image_out_path="/outputs/",

    #             do_pacthes=True,
    #             tile=320,
    #             tile_overlap=128+64)

    # end = time.time()
    # print(f"Completed in {end-st}s")


    # #----------------------------------------------------------------------------------------------------------
    # # Deblur, BSD
    # #90Kmodel
    # config = "options/Turtle_Derain_VRDS.yml"
    # model_path = "trained_models/BSD.pth"
    # model_name = "Turtle_BSD082_simple_320_128"
    # print(model_name)
    # _, _ = main(model_path=model_path,
    #             model_name=model_name, 
    #             config_file=config,

    #             dataset_name="BSD", #GoPro, SR, NightRain, DVD, Set8
    #             task_name="Deblurring", #Deblurring, SR, Deraining, Deblurring, Denoising

    #             model_type="t0",

    #             save_image=True,
    #             image_out_path="/outputs/",

    #             do_pacthes=True,
    #             tile=320,
    #             tile_overlap=128)

    # end = time.time()
    # print(f"Completed in {end-st}s")
