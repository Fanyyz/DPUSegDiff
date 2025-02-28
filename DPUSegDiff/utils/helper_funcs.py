from matplotlib import pyplot as plt
import numpy as np
import cv2
import yaml
from termcolor import colored
from src.utils.header import isfunction
import os
import json
import torch
from scipy import ndimage
from skimage import feature
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from matplotlib import gridspec


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class _bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def cprint(string, p=None):
    if not p:
        print(string)
        return
    pre = f"{_bcolors.ENDC}"

    if "bold" in p.lower():
        pre += _bcolors.BOLD
    elif "underline" in p.lower():
        pre += _bcolors.UNDERLINE
    elif "header" in p.lower():
        pre += _bcolors.HEADER

    if "warning" in p.lower():
        pre += _bcolors.WARNING
    elif "error" in p.lower():
        pre += _bcolors.FAIL
    elif "ok" in p.lower():
        pre += _bcolors.OKGREEN
    elif "info" in p.lower():
        if "blue" in p.lower():
            pre += _bcolors.OKBLUE
        else:
            pre += _bcolors.OKCYAN

    print(f"{pre}{string}{_bcolors.ENDC}")


_print = cprint


def load_config(config_filepath):
    try:
        with open(config_filepath, "r") as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        _print(f"Config file not found! <{config_filepath}>", "error_bold")
        exit(1)


def show_sbs(im1, im2, figsize=[8, 4], im1_title="Image", im2_title="Mask", show=True):
    if im1.shape[0] < 4:
        im1 = np.array(im1)
        im1 = np.transpose(im1, [1, 2, 0])

    if im2.shape[0] < 4:
        im2 = np.array(im2)
        im2 = np.transpose(im2, [1, 2, 0])

    _, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].imshow(im1)
    axs[0].set_title(im1_title)
    axs[1].imshow(im2, cmap="gray")
    axs[1].set_title(im2_title)
    if show:
        plt.show()


def get_model_path(name, dir="./", **kwargs):
    items = [
        name,
    ]
    
    file_name = "_".join(items)
    extension = "_best.pth"

    path = os.path.join(dir, file_name + extension)
    return path


def save_test_res(DT, img, msk, generated_msk, fp):
    fig = plt.figure(figsize=(10, 10))  # Notice the equal aspect ratio
    ax = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_aspect("equal")
    fig.subplots_adjust(wspace=0, hspace=0.1)

    plt.subplot(221)
    plt.imshow(DT.get_reverse_transform_to_pil()(img))
    plt.title("img")
    plt.axis("off")
    plt.subplot(222)
    plt.imshow(DT.get_reverse_transform_to_numpy()(msk)[:, :, 0])
    plt.title("msk")
    plt.axis("off")
    plt.subplot(223)
    plt.imshow(DT.get_reverse_transform_to_numpy()(generated_msk)[:, :, 0])
    plt.title(f"generated msk")
    plt.axis("off")
    plt.subplot(224)
    plt.imshow(DT.get_reverse_transform_to_binary()(generated_msk)[:, :, 0])
    plt.title(f"generated binary msk")
    plt.axis("off")
    plt.savefig(fp, bbox_inches="tight")
    plt.close()


def print_config(config, logger=None):
    conf_str = json.dumps(config, indent=2)
    if logger:
        logger.info(f"\n{' Config '.join(2*[10*'>>',])}\n{conf_str}\n{28*'>>'}")
    else:
        _print("Config:", "info_underline")
        print(conf_str)
        print(30 * "~-", "\n")
        


def get_conf_name(config):
    items = []
    items.append(f'n-{config["model"]["name"]}')
    items.append(f's-{config["dataset"]["input_size"]}')
    items.append(f'b-{config["data_loader"]["train"]["batch_size"]}')
    items.append(f't-{config["diffusion"]["schedule"]["timesteps"]}')
    #     items.append(f'op-{config["training"]["optimizer"]["name"].lower()}')
    #     items.append(f'lr-{config["training"]["optimizer"]["params"]["lr"]}')
    #     items.append(f'ep-{config["training"]["epochs"]}')
    items.append(f'sc-{config["diffusion"]["schedule"]["mode"]}')
    return "_".join(items)


# ----------------------------------------------------------------
# weight calculations
# ----------------------------------------------------------------


# (batch, 1, image_size, image_size)
def cal_gamma_by_timesteps(
    timesteps: torch.Tensor, image_size: int, min, max, device="cpu"
) -> torch.Tensor:
    # h=self.gamma[timesteps]
    # print(f"timesteps: {timesteps}")
    # print(f"gammas: {h}")
    gammas = torch.flip(
        torch.linspace(min, max, 1000).to(device),
        (0,),
    )
    return gammas[timesteps].view(-1, 1, 1, 1).expand(-1, -1, image_size, image_size)


def canny_edge(batches: torch.Tensor) -> list:
    batch_size = batches.shape[0]
    batch = ((batches[:, 1, :, :] + 1) / 2).detach().cpu().numpy()
    edges = [
        feature.canny(batch[i], sigma=1) for i in range(batch_size)
    ]  # apply Canny edge detection
    return edges


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    # Function definition to normalize input tensor along height and width
    # Input:
    #   - tensor: torch.Tensor of size  (*other, height, width)
    # Output:
    #   - normalized_tensor: torch.Tensor of size (*other, height, width),
    #                        which is the result of normalization
    """
    min_vals = torch.min(tensor, dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]

    max_vals = torch.max(tensor, dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    normalized_tensor = (tensor - min_vals) / (max_vals - min_vals)
    return normalized_tensor


def custom_normalize(t: torch.Tensor, min_w: float, max_w: float) -> torch.Tensor:
    """
    This function customizes the normalization of a given tensor.

    Args:
    t (torch.Tensor): Input tensor.
    min_w (float): Minimum value for output tensor
    max_w (float): Maximum value for output tensor

    Returns:
    The normalized tensor

    """
    tensor_min = t.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    t = t - tensor_min
    t = (t / t.sum(dim=(-1, -2), keepdim=True)) * t.shape[-1] ** 2
    tensor = normalize(t) * (max_w - min_w) + min_w
    return tensor


def standardize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(tensor, dim=(-1, -2), keepdim=True)
    std = torch.std(tensor, dim=(-1, -2), keepdim=True)
    return (tensor - mean) / std


def cal_weights_using_neg(
    masks: torch.Tensor,
    timesteps: torch.Tensor,
    min_w: float,
    max_w: float,
    min_g=0.2,
    max_g=1.75,
    device="cpu",
) -> torch.Tensor:

    edges = canny_edge(masks)

    distance_maps = [
        torch.from_numpy(ndimage.distance_transform_edt(np.logical_not(edge)))
        for edge in edges
    ]

    distance_batches = torch.stack(distance_maps, axis=0).to(device).unsqueeze(1)

    gammas = cal_gamma_by_timesteps(
        timesteps, distance_batches.shape[-1], min_g, max_g, device=device
    )
    temp = -distance_batches + (masks.shape[-1] * np.sqrt(2))

    weights = torch.pow(normalize(torch.sqrt(temp)), gammas)
    weights_shifted = custom_normalize(weights, min_w, max_w)
    #weights_shifted += 1

    return (
        weights_shifted,
        torch.stack([torch.from_numpy(edge) for edge in edges], axis=0)
        .to(device)
        .unsqueeze(1),
        distance_batches,
    )


def cal_weights_using_inv(
    masks: torch.Tensor, gamma: torch.Tensor, min: float, max: float, device="cpu"
) -> torch.Tensor:
    edges = canny_edge(masks)

    distance_maps = [
        torch.from_numpy(ndimage.distance_transform_edt(np.logical_not(edge)))
        for edge in edges
    ]

    distance_batches = torch.stack(distance_maps, axis=0).to(device).unsqueeze(1)
    weights = (1.0 / (distance_batches + 1)) ** gamma
    weights = torch.log(weights + 1)
    weights_norm = normalize(weights)

    weights_shifted = weights_norm * (max - min) + min

    return (
        weights_shifted,
        torch.stack([torch.from_numpy(edge) for edge in edges], axis=0)
        .to(device)
        .unsqueeze(1),
        distance_batches,
    )


def calc_edge(x, mode='canny'):
    x = np.uint8(x)
    edge = cv2.Canny(image=x, threshold1=0, threshold2=0)
    return edge

from copy import deepcopy
def draw_boundary(x, img, color):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = (img-img.min())/(img.max()-img.min())

    edged = calc_edge(x[0,:], mode='canny')
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for idx, imc in enumerate(range(3)):
        img[idx,:,:] = cv2.drawContours(np.uint8(img[idx,:,:]*255), contours, -1, color[idx], 1)/255.
    return img


def calc_distance_map(x, mode='l2'):
    # if isinstance(x, torch.Tensor):
    #     x = x.numpy()
    
    # Convert the data to grayscale [0,255]
    binary_x = 1 - np.uint8((x-x.min())/(x.max()-x.min()))

    if mode.lower() == 'l1':
        dt_mode = cv2.DIST_L1
    elif mode.lower() == 'l2':
        dt_mode = cv2.DIST_L2
    else:
        raise ValueError("<mode> must be 'l1' or 'l2'.")
    
    # Calculate the distance transform
    dist_transform= cv2.distanceTransform(binary_x, dt_mode, 0)

    return dist_transform


def calc_boundary_att(batch_x, batch_t, T, gamma=1.5, *args, **kwargs):
    """
    Parameters:
        - batch_x : [tensor] |-> input data matrix
        - batch_t : [tensor] |-> current timestep
        - T       : [int]    |-> maximum timesteps
        - gamma   : [float]  |-> sharpness [default is 1.5]
    Output:
        - boundary_att
    """
    
    
    # boundary thickness (max value thickness)
    bt = np.round(batch_x.shape[-1]*0.01)
    
    X = batch_x.detach().cpu().numpy()
    X = (X-X.min())/(X.max()-X.min()) # normalize because X is in range ~ (-1, 1)
    device = batch_x.get_device()
    
    atts = []
    for x, t in zip(X, batch_t):
        if x.sum().item() > X.shape[-1]**2/100.: # foreground area is bigger than 1% of the image
            x = (x-x.min())/(x.max()-x.min())
            edge = calc_edge(x[0,:])
            dist_x = calc_distance_map(edge, mode='l2')
            tmp = X.shape[-1]*1.1415 - dist_x
            normalized_inv_dist_x = (tmp-tmp.min())/ (tmp.max()-tmp.min())
            
            t_p = ((gamma*(T-t.item()))/T)**gamma
            att = normalized_inv_dist_x**t_p            
        else:
            att = np.ones_like(x[0,:])
        atts.append(att)

    atts = np.array(atts)
    
    W = torch.stack([torch.from_numpy(att) for att in atts], dim=0)
    if device != -1:
        W = W.to(device)

    W = torch.unsqueeze(W, dim=1)
    return W

# def calc_boundary_att(batch_x, max_bp=0.2, *args, **kwargs):
#     """
#     Parameters:
#         - batch_x: input data matrix
#         - max_bp: maximum percent of boundary margin coresponding to input size. Default is `0.2`.
#     Output:
#         - boundary_att
#     """
    
#     # max boundary margin
#     max_dist = np.round(batch_x.shape[-1]*max_bp)
    
#     # boundary thickness (max value thickness)
#     bt = np.round(batch_x.shape[-1]*0.01)
    
#     X = batch_x.detach().cpu().numpy()
#     device = batch_x.get_device()
    
#     atts = []
#     for x in X:
#         if x.sum().item() > 50:
#             x = (x-x.min())/(x.max()-x.min())
#             edge = calc_edge(x[0,:])
#             dist_x = calc_distance_map(edge, mode='l2')
#             inv_dist_x = max_dist - dist_x
#             clipped_inv_dist_x = np.clip(inv_dist_x, 0, max_dist-bt)
#             tmp = clipped_inv_dist_x
#             normalized_inv_dist_x = (tmp-tmp.min())/ (tmp.max()-tmp.min())
#             # att = (clipped_inv_dist_x/255.)
#             att = normalized_inv_dist_x
#         else:
#             att = np.ones_like(x[0,:])
#         atts.append(att)

#     atts = np.array(atts)
    
#     # save_makegrid_img(X[:,0], atts, nrow=2)
#     # exit()
    
#     W = torch.stack([torch.from_numpy(att) for att in atts], dim=0)
#     if device != -1:
#         W = W.to(device)

#     W = torch.unsqueeze(W, dim=1)
#     return W


def save_makegrid_img(Xs, Ys, nrow=2):
    reses = []
    for x, y in zip(Xs, Ys):

        x = (x - x.min()) / (x.max() - x.min())
        
        reses.append(x)
        reses.append(y)
    
    t = torch.tensor(reses)
    t = torch.unsqueeze(t, 1)
    # t = torch.movedim(t, -1, 1)

    grid = torchvision.utils.make_grid(t, nrow=nrow)
    grid = torch.movedim(grid, 0, -1)

    print(grid.shape)
    
    cv2.imwrite('./atts.png', np.uint8(grid.numpy()*255))
    
    # plt.figure(figsize=(10, nrow*10))
    # plt.imshow(grid)
    # plt.tight_layout()
    # plt.axis('off')
    
    # plt.savefig('./atts.png')
    
def mean_of_list_of_tensors(list_of_tensors):
    mean_x = torch.zeros_like(list_of_tensors[0])
    for t in list_of_tensors:
        mean_x += t
    mean_x /= len(list_of_tensors)
    return mean_x


def variance_of_list_of_tensors(list_of_tensors):
    mean_x = torch.zeros_like(list_of_tensors[0])  # 初始化均值张量
    for t in list_of_tensors:
        mean_x += t  # 累加每个张量
    mean_x /= len(list_of_tensors)  # 计算均值

    var_x = torch.zeros_like(list_of_tensors[0])  # 初始化方差张量
    for t in list_of_tensors:
        var_x += (t - mean_x) ** 2  # 累加每个张量与均值差的平方

    var_x /= len(list_of_tensors)  # 计算方差
    return var_x



from torchvision.utils import save_image


# def save_image_with_colorbar(image, save_path, vmin, vmax):
#     # 确保将张量移动到 CPU 并转换为 numpy 数组
#     image = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
#
#     # 创建一个新的图形
#     plt.figure(figsize=(3, 2),dpi=64)
#
#     # 显示图像，并设置颜色范围
#     im = plt.imshow(image, cmap='jet', vmin=vmin, vmax=vmax)
#
#     # 添加颜色条
#     plt.colorbar(im)
#
#     # 去掉坐标轴
#     plt.axis('off')
#
#     # 保存图像
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#     plt.close()

def save_image_with_colorbar(image, save_path, vmin, vmax):
    # 确保将张量移动到 CPU 并转换为 numpy 数组
    image = image.cpu().numpy() if isinstance(image, torch.Tensor) else image

    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=(2, 2), dpi=800)  # 图像区域大小调整为 128×128 像素
    ax.set_position([0.1, 0.1, 0.65, 0.8])  # 调整主图的轴位置

    # 显示图像，并设置颜色范围
    im = ax.imshow(image, cmap='jet', vmin=vmin, vmax=vmax)

    # 添加颜色条并设置颜色条长度与图像一致
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format='%.2f')
    cbar.ax.tick_params(labelsize=6)  # 调整颜色条字体大小

    # 去掉坐标轴
    ax.axis('off')

    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_sampling_results_as_imgs(
    batch_imgs, batch_msks, batch_ids, batch_prds,sample1,sample2,batch_variance,
    ensemble_list_of_samples_in_timesteps=None, middle_steps_of_sampling=4,
    save_dir="./saved_imgs",
    dataset_name=None,
    result_id=None,
    img_ext="png",
    save_mat=False
):
    save_directory =  "/".join([d for d in [save_dir, dataset_name, result_id] if d])
    
    # check dir
    Path(save_directory).mkdir(exist_ok=True, parents=True)

    for im, gt, id, pd,s1,s2,variance in zip(batch_imgs, batch_msks, batch_ids, batch_prds,sample1,sample2,batch_variance):
        sd = f"{save_directory}/{id}"
        Path(sd).mkdir(exist_ok=True)
        
        save_image(im   , f"{sd}/im.{img_ext}")
        save_image(gt[0], f"{sd}/gt.{img_ext}")
        save_image(pd[0], f"{sd}/pd.{img_ext}")

        save_image(s1[0], f"{sd}/s1.{img_ext}")
        save_image(s2[0], f"{sd}/s2.{img_ext}")
        save_image_with_colorbar(pd[0], f"{sd}/mean.{img_ext}", vmin=0, vmax=1)
        save_image_with_colorbar(variance[0], f"{sd}/variance.{img_ext}", vmin=0, vmax=0.25)
        
        # draw gt-pd on the image
        db = draw_boundary(torch.where(gt>0, 1, 0), im, (0, 255, 0))
        db = draw_boundary(torch.where(pd>0, 1, 0), db, (0, 0, 255))
        save_image(torch.tensor(db), f"{sd}/db.{img_ext}")

        if save_mat:
            torch.save(gt[0], f"{sd}/gt.pt")
            torch.save(pd[0], f"{sd}/pd.pt")
            
    if ensemble_list_of_samples_in_timesteps:
        T = len(ensemble_list_of_samples_in_timesteps[0])
        sample_ts = np.int32(np.round(np.linspace(0, T-1, middle_steps_of_sampling+2))) # added 2 because inxed 0 and T-1
        for i, ensemble_i in enumerate(ensemble_list_of_samples_in_timesteps):
            batch_samples = [ensemble_i[i] for i in sample_ts]
            for batch_sample, t in zip(batch_samples, sample_ts):
                for sample, id in zip(batch_sample, batch_ids):
                    save_image(sample[0], f"{save_directory}/{id}/t_{t}.{img_ext}")


def save_sampling_multiresults_as_imgs(
        batch_imgs, batch_msks, batch_ids, batch_prds,
        ensemble_list_of_samples_in_timesteps=None, middle_steps_of_sampling=4,
        save_dir="./saved_imgs",
        dataset_name=None,
        result_id=None,
        img_ext="png",
        save_mat=False
):
    save_directory = "/".join([d for d in [save_dir, dataset_name, result_id] if d])

    # 检查目录是否存在
    Path(save_directory).mkdir(exist_ok=True, parents=True)

    for im, gt, id, pd in zip(batch_imgs, batch_msks, batch_ids, batch_prds):
        sd = f"{save_directory}/{id}"
        Path(sd).mkdir(exist_ok=True)

        # 保存原始图像
        save_image(im, f"{sd}/im.{img_ext}")
        # brfore1 = pd.cpu().numpy()
        # brfore2 = gt.cpu().numpy()
        # after1 = pd.cpu().numpy()

        # 绘制分割结果（视盘+视杯）
        combined_gt = create_colored_segmentation(gt)
        combined_pd = create_colored_segmentation(pd)

        save_image(combined_gt, f"{sd}/gt.{img_ext}")
        save_image(combined_pd, f"{sd}/pd.{img_ext}")

        # 在原图上绘制边界
        db = draw_boundary(torch.where(gt[0] > 0, 1, 0), im, (0, 255, 0))  # 视盘
        db = draw_boundary(torch.where(gt[1] > 0, 1, 0), db, (255, 0, 0))  # 视杯
        save_image(torch.tensor(db), f"{sd}/db.{img_ext}")

        if save_mat:
            torch.save(gt, f"{sd}/gt.pt")
            torch.save(pd, f"{sd}/pd.pt")

    if ensemble_list_of_samples_in_timesteps:
        T = len(ensemble_list_of_samples_in_timesteps[0])
        sample_ts = np.int32(np.round(np.linspace(0, T - 1, middle_steps_of_sampling + 2)))  # 包含索引0和T-1
        for i, ensemble_i in enumerate(ensemble_list_of_samples_in_timesteps):
            batch_samples = [ensemble_i[i] for i in sample_ts]
            for batch_sample, t in zip(batch_samples, sample_ts):
                for sample, id in zip(batch_sample, batch_ids):
                    save_image(sample[0], f"{save_directory}/{id}/t_{t}.{img_ext}")


def create_colored_segmentation(segmentation):
    """
    根据分割结果创建彩色图像：
    背景：黑色；视盘：绿色；视杯：红色
    """
    h, w = segmentation.shape[1], segmentation.shape[2]
    colored_seg = torch.zeros(3, h, w)  # 创建彩色通道的空白图像 (C, H, W)

    # 视盘（绿色通道）
    colored_seg[1] = segmentation[1]
    # 视杯（红色通道）
    colored_seg[0] = segmentation[2]

    colored_seg[2] = segmentation[3]

    return colored_seg

def process_preds(preds):
    """
    对网络输出的预测值进行后处理：
    - 背景：值为0
    - 前景（视盘/视杯）：值为1
    """
    # 假设网络输出是概率图 (b, 2, h, w)，需要进行阈值化
    threshold = 0.5
    binary_preds = (preds > threshold).float()  # 二值化
    return binary_preds