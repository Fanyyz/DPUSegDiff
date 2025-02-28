from utils.header import (
    torch,  # 导入 PyTorch 库
    nn,  # 导入 PyTorch 的神经网络模块
    math,  # 导入 Python 标准库中的数学模块
)

# 正弦位置编码类，用于生成时间步的编码
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        """
        初始化 SinusoidalPositionEmbeddings。
        :param dim: 位置编码的维度大小（embedding size）
        """
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        self.dim = dim  # 保存位置编码的维度

    def forward(self, time):
        """
        前向传播函数，生成时间步的正弦位置编码。
        :param time: 时间步张量，形状为 (batch_size, )
        :return: 正弦位置编码张量，形状为 (batch_size, dim)
        """
        device = time.device  # 获取输入的时间步张量所在的设备（CPU/GPU）
        half_dim = self.dim // 2  # 将维度减半，用于分别计算 sin 和 cos 编码

        # 计算频率因子。log(10000) / (half_dim - 1) 确保编码覆盖宽广的频率范围。
        embeddings = math.log(10000) / (half_dim - 1)

        # 生成一个 [0, 1, 2, ..., half_dim-1] 的序列，乘以频率因子并取指数以得到不同频率
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # 将时间步 `time` 扩展为 (batch_size, 1) 的形状，再与生成的频率进行广播相乘
        embeddings = time[:, None] * embeddings[None, :]

        # 使用 sin 和 cos 函数对频率乘积进行编码，并在最后一维上进行拼接
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings  # 返回生成的正弦位置编码
