import torch
from tqdm import tqdm
from models import *


@torch.no_grad()  # 禁用梯度计算，提高推理效率，节省内存
def p_sample(forward_schedule, model, images, x, t, t_index):
    # 获取扩散过程相关的系数
    fs = forward_schedule
    exs = fs.extract(t, x.shape)  # 根据当前时间步 t 提取对应参数
    betas_t = exs["betas"]  # 时间步 t 对应的 beta 系数
    sqrt_one_minus_alphas_cumprod_t = exs["sqrt_one_minus_alphas_cumprod"]  # 时间步 t 累积 alpha 的平方根
    sqrt_recip_alphas_t = exs["sqrt_recip_alphas"]  # 时间步 t alpha 的倒数平方根

    # 使用模型（噪声预测器）预测噪声，DermoSegDiff 和 Baseline 是不同的模型
    if isinstance(model, V9):
        predicted_noise = model(x, images, t)  # DermoSegDiff 模型使用三个输入
    elif isinstance(model, Baseline):
        predicted_noise = model(x, t, images)  # Baseline 模型使用不同顺序的输入
    else:
        NotImplementedError('given model is unknown!')  # 抛出异常，未实现的模型类型

    # 计算模型预测的去噪后图像的均值（扩散过程反向步骤），对应论文的公式 11
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )

    # 如果是最后一个时间步（t_index == 0），直接返回均值
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = exs["posterior_variance"]  # 提取后验方差
        noise = torch.randn_like(x)  # 生成与输入形状相同的高斯噪声
        # 在非最后一步时，返回加上噪声的模型预测结果
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# 采样循环（逐步还原图像），包括返回所有时间步的图像
@torch.no_grad()
def p_sample_loop(forward_schedule, model, images, out_channels, desc=""):
    timesteps = forward_schedule.timesteps  # 获取总的时间步数
    device = next(model.parameters()).device  # 获取模型设备（CPU/GPU）

    # 初始化掩码图像的形状和大小，out_channels 表示输出通道数
    shape = (images.shape[0], out_channels, images.shape[2], images.shape[3])
    b = shape[0]

    # 从纯噪声开始（为每个批次样本初始化一个随机噪声掩码）
    msk = torch.randn(shape, device=device)
    msks = []  # 用于保存每个时间步的掩码

    # 逐步逆扩散还原图像，每个时间步减少噪声
    desc = f"{desc} - sampling loop time step" if desc else "sampling loop time step"
    for i in tqdm(
            reversed(range(0, timesteps)), desc=desc, total=timesteps, leave=False
    ):
        # 在每个时间步调用 p_sample 函数，逐步去噪
        msk = p_sample(
            forward_schedule,
            model,
            images,
            msk,
            torch.full((b,), i, device=device, dtype=torch.long),  # 当前时间步 t
            i,
        )
        msks.append(msk.detach().cpu())  # 将当前时间步的去噪掩码保存到列表中
    return msks  # 返回所有时间步的掩码


# 主采样函数，用于从模型中生成采样
@torch.no_grad()
def sample(forward_schedule, model, images, out_channels=2, desc=None):
    return p_sample_loop(forward_schedule, model, images, out_channels, desc)  # 调用 p_sample_loop 逐步生成采样


# 使用噪声反推原始输入，模拟扩散过程的逆过程
def reverse_by_epsilon(forward_process, predicted_noise, x, t):
    fs = forward_process.forward_schedule  # 提取前向扩散过程中的时间步参数
    exs = fs.extract(t, x.shape)

    betas_t = exs["betas"]
    sqrt_one_minus_alphas_cumprod_t = exs["sqrt_one_minus_alphas_cumprod"]
    sqrt_recip_alphas_t = exs["sqrt_recip_alphas"]
    posterior_variance_t = exs["posterior_variance"]

    # 使用扩散模型的逆向公式，从噪声中还原原始输入
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )

    noise = torch.randn_like(x) if t[0].item() > 0 else 0  # 如果 t > 0，添加高斯噪声，否则没有噪声
    # 返回计算后的均值加上噪声，模拟扩散逆向过程
    res = model_mean + torch.sqrt(posterior_variance_t) * noise

    return res  # 返回结果
