# Pytorch 基础学习笔记

> 本文档是一份 PyTorch 学习指南手册，共 16 章：从张量的创建、运算、索引与形状，到自动微分与模型保存加载，再到神经网络模块（torch.nn）、函数库（torch.nn.functional）、设备与内存管理、进阶实用技巧，最后是昇腾 NPU 上的 torch 适配专题（设备与内存、融合算子与布局、图模式与降级策略）。每章配有可独立运行的示例与真实输出结果。

## 目录

1. [1. 张量的创建](#1-张量的创建)
2. [2. 张量数值计算](#2-张量数值计算)
3. [3. 张量类型转换](#3-张量类型转换)
4. [4. 张量拼接操作](#4-张量拼接操作)
5. [5. 张量索引操作](#5-张量索引操作)
6. [6. 张量形状操作](#6-张量形状操作)
7. [7. 张量运算函数](#7-张量运算函数)
8. [8. 自动微分模块](#8-自动微分模块)
9. [9. 模型的保存与加载](#9-模型的保存与加载)
10. [10. torch.nn 神经网络模块](#10-torchnn-神经网络模块)
11. [11. torch.nn.functional 函数库](#11-torchnnfunctional-函数库)
12. [12. 设备、数据类型与内存管理](#12-设备数据类型与内存管理)
13. [13. 进阶实用技巧](#13-进阶实用技巧)
14. [14. torch 与 NPU：设备与内存管理](#14-torch-与-npu设备与内存管理)
15. [15. torch 与 NPU：融合算子与布局适配](#15-torch-与-npu融合算子与布局适配)
16. [16. torch 与 NPU：图模式、分布式与降级策略](#16-torch-与-npu图模式分布式与降级策略)

---

# 1. 张量的创建

掌握张量创建。


PyTorch 是一个 Python 深度学习框架，它将数据封装成张量（Tensor）来进行运算。PyTorch 中的张量就是元素为同一种数据类型的多维矩阵。在 PyTorch 中，张量以 "类" 的形式封装起来，对张量的一些运算、处理的方法被封装在类中。

## 基本创建方式

1. torch.tensor 根据指定数据创建张量
2. torch.Tensor 根据形状创建张量, 其也可用来创建指定数据的张量
3. torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量

```python
import torch
import numpy as np
import random

# 1. 根据已有数据创建张量
def test01():

    # 1. 创建张量标量
    data = torch.tensor(10)
    print(data)

    # 2. numpy 数组, 由于 data 为 float64, 下面代码也使用该类型
    data = np.random.randn(2, 3)
    data = torch.tensor(data)
    print(data)

    # 3. 列表, 下面代码使用默认元素类型 float32
    data = [[10., 20., 30.], [40., 50., 60.]]
    data = torch.tensor(data)
    print(data)


# 2. 创建指定形状的张量
def test02():

    # 1. 创建2行3列的张量, 默认 dtype 为 float32
    data = torch.Tensor(2, 3)
    print(data)

    # 2. 注意: 如果传递列表, 则创建包含指定元素的张量
    data = torch.Tensor([10])
    print(data)

    data = torch.Tensor([10, 20])
    print(data)


# 3. 使用具体类型的张量
def test03():

    # 1. 创建2行3列, dtype 为 int32 的张量
    data = torch.IntTensor(2, 3)
    print(data)

    # 2. 注意: 如果传递的元素类型不正确, 则会进行类型转换
    data = torch.IntTensor([2.5, 3.3])
    print(data)

    # 3. 其他的类型
    data = torch.ShortTensor()  # int16
    data = torch.LongTensor()   # int64
    data = torch.FloatTensor()  # float32
    data = torch.DoubleTensor() # float64


if __name__ == '__main__':
    test02()
```

程序输出结果:

```python
tensor(10)
tensor([[ 0.1345,  0.1149,  0.2435],
        [ 0.8026, -0.6744, -1.0918]], dtype=torch.float64)
tensor([[10., 20., 30.],
        [40., 50., 60.]])
tensor([[0.0000e+00, 3.6893e+19, 2.2018e+05],
        [4.6577e-10, 2.4158e-12, 1.1625e+33]])
tensor([10.])
tensor([10., 20.])
tensor([[         0, 1610612736, 1213662609],
        [ 805308409,  156041223,          1]], dtype=torch.int32)
tensor([2, 3], dtype=torch.int32)
```

## 创建线性和随机张量

1. torch.arange 和 torch.linspace 创建线性张量
2. torch.random.init\_seed 和 torch.random.manual\_seed 随机种子设置
3. torch.randn 创建随机张量

```python
import torch


# 1. 创建线性空间的张量
def test01():

    # 1. 在指定区间按照步长生成元素 [start, end, step)
    data = torch.arange(0, 10, 2)
    print(data)

    # 2. 在指定区间按照元素个数生成
    data = torch.linspace(0, 11, 10)
    print(data)


# 2. 创建随机张量
def test02():

    # 1. 创建随机张量
    data = torch.randn(2, 3)  # 创建2行3列张量
    print(data)

    # 2. 随机数种子设置
    print('随机数种子:', torch.random.initial_seed())
    torch.random.manual_seed(100)
    print('随机数种子:', torch.random.initial_seed())


if __name__ == '__main__':
    test02()
```

程序输出结果:

```python
tensor([0, 2, 4, 6, 8])
tensor([ 0.0000,  1.2222,  2.4444,  3.6667,  4.8889,  6.1111,  7.3333,  8.5556,
         9.7778, 11.0000])
tensor([[-0.5209, -0.2439, -1.1780],
        [ 0.8133,  1.1442,  0.6790]])
随机数种子: 4508475192273306739
随机数种子: 100
```

## 创建01张量

1. torch.ones 和 torch.ones\_like 创建全1张量
2. torch.zeros 和 torch.zeros\_like 创建全0张量
3. torch.full 和 torch.full\_like 创建全为指定值张量

```python
import torch


# 1. 创建全0张量
def test01():

    # 1. 创建指定形状全0张量
    data = torch.zeros(2, 3)
    print(data)

    # 2. 根据张量形状创建全0张量
    data = torch.zeros_like(data)
    print(data)


# 2. 创建全1张量
def test02():

    # 1. 创建指定形状全0张量
    data = torch.ones(2, 3)
    print(data)

    # 2. 根据张量形状创建全0张量
    data = torch.ones_like(data)
    print(data)


# 3. 创建全为指定值的张量
def test03():

    # 1. 创建指定形状指定值的张量
    data = torch.full([2, 3], 10)
    print(data)

    # 2. 根据张量形状创建指定值的张量
    data = torch.full_like(data, 20)
    print(data)


if __name__ == '__main__':
    test01()
    test02()
    test03()
```

程序输出结果:

```python
tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[10, 10, 10],
        [10, 10, 10]])
tensor([[20, 20, 20],
        [20, 20, 20]])
```

## 张量元素类型转换

1. tensor.type(torch.DoubleTensor)
2. torch.double()

```python
import torch


def test():

    data = torch.full([2, 3], 10)
    print(data.dtype)

    # 将 data 元素类型转换为 float64 类型

    # 1. 第一种方法
    data = data.type(torch.DoubleTensor)
    print(data.dtype)

    # 转换为其他类型
    # data = data.type(torch.ShortTensor)
    # data = data.type(torch.IntTensor)
    # data = data.type(torch.LongTensor)
    # data = data.type(torch.FloatTensor)

    # 2. 第二种方法
    data = data.double()
    print(data.dtype)

    # 转换为其他类型
    # data = data.short()
    # data = data.int()
    # data = data.long()
    # data = data.float()


if __name__ == '__main__':
    test()
```

程序输出结果:

```python
torch.int64
torch.float64
torch.float64
```

---


# 2. 张量数值计算

掌握张量基本运算、阿达玛积、点积运算与指定运算设备。


PyTorch 计算的数据都是以张量形式存在, 我们需要掌握张量各种运算. 并且, 我们可以在 CPU 中运算, 也可以在 GPU 中运算.

## 张量基本运算

基本运算中，包括 add、sub、mul、div、neg 等函数, 以及这些函数的带下划线的版本 add\_、sub\_、mul\_、div\_、neg\_，其中带下划线的版本为修改原数据。

```python
import numpy as np
import torch


def test():

    data = torch.randint(0, 10, [2, 3])
    print(data)
    print('-' * 50)

    # 1. 不修改原数据
    new_data = data.add(10)  # 等价 new_data = data + 10
    print(new_data)
    print('-' * 50)

    # 2. 直接修改原数据
    # 注意: 带下划线的函数为修改原数据本身
    data.add_(10)  # 等价 data += 10
    print(data)

    # 3. 其他函数
    print(data.sub(100))
    print(data.mul(100))
    print(data.div(100))
    print(data.neg())


if __name__ == '__main__':
    test()
```

程序输出结果:

```python
tensor([[3, 7, 4],
        [0, 0, 6]])
--------------------------------------------------
tensor([[13, 17, 14],
        [10, 10, 16]])
--------------------------------------------------
tensor([[13, 17, 14],
        [10, 10, 16]])
tensor([[-87, -83, -86],
        [-90, -90, -84]])
tensor([[1300, 1700, 1400],
        [1000, 1000, 1600]])
tensor([[0.1300, 0.1700, 0.1400],
        [0.1000, 0.1000, 0.1600]])
tensor([[-13, -17, -14],
        [-10, -10, -16]])
```

## 阿达玛积

阿达玛积指的是矩阵对应位置的元素相乘.

```python
import numpy as np
import torch


def test():

    data1 = torch.tensor([[1, 2], [3, 4]])
    data2 = torch.tensor([[5, 6], [7, 8]])

    # 第一种方式
    data = torch.mul(data1, data2)
    print(data)
    print('-' * 50)

    # 第二种方式
    data = data1 * data2
    print(data)
    print('-' * 50)


if __name__ == '__main__':
    test()
```

程序输出结果:

```python
tensor([[ 5, 12],
        [21, 32]])
--------------------------------------------------
tensor([[ 5, 12],
        [21, 32]])
--------------------------------------------------
```

## 点积运算

点积运算要求第一个矩阵 shape: (n, m)，第二个矩阵 shape: (m, p), 两个矩阵点积运算 shape 为: (n, p)。

1. 运算符 @ 用于进行两个矩阵的点乘运算
2. torch.mm 用于进行两个矩阵点乘运算, 要求输入的矩阵为2维
3. torch.bmm 用于批量进行矩阵点乘运算, 要求输入的矩阵为3维
4. torch.matmul 对进行点乘运算的两矩阵形状没有限定.
   1. 对于输入都是二维的张量相当于 mm 运算.
   2. 对于输入都是三维的张量相当于 bmm 运算
   3. 对数输入的 shape 不同的张量, 对应的最后几个维度必须符合矩阵运算规则

```python
import numpy as np
import torch


# 1. 点积运算
def test01():

    data1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    data2 = torch.tensor([[5, 6], [7, 8]])

    # 第一种方式
    data = data1 @ data2
    print(data)
    print('-' * 50)

    # 第二种方式
    data = torch.mm(data1, data2)
    print(data)
    print('-' * 50)

    # 第三种方式
    data = torch.matmul(data1, data2)
    print(data)
    print('-' * 50)


# 2. torch.mm 和 torch.matmull 的区别
def test02():

    # matmul 可以两个维度可以不同
    # 第一个张量: (3, 4, 5)
    # 第二个张量: (5, 4)
    # torch.mm 不可以相乘，而 matmul 则可以相乘

    print(torch.matmul(torch.randn(3, 4, 5), torch.randn(5, 4)).shape)
    print(torch.matmul(torch.randn(5, 4), torch.randn(3, 4, 5)).shape)


# 3. torch.mm 函数的用法
def test03():

    # 批量点积运算
    # 第一个维度为 batch_size
    # 矩阵的二三维要满足矩阵乘法规则

    data1 = torch.randn(3, 4, 5)
    data2 = torch.randn(3, 5, 8)

    data = torch.bmm(data1, data2)
    print(data.shape)


if __name__ == '__main__':
    test01()
    test02()
    test03()
```

程序输出结果:

```python
tensor([[19, 22],
        [43, 50],
        [67, 78]])
--------------------------------------------------
tensor([[19, 22],
        [43, 50],
        [67, 78]])
--------------------------------------------------
tensor([[19, 22],
        [43, 50],
        [67, 78]])
--------------------------------------------------
torch.Size([3, 4, 4])
torch.Size([3, 5, 5])
torch.Size([3, 4, 8])
```

## 指定运算设备

PyTorch 默认会将张量创建在 CPU 控制的内存中, 即: 默认的运算设备为 CPU。我们也可以将张量创建在 GPU 上, 能够利用对于矩阵计算的优势加快模型训练。将张量移动到 GPU 上有两种方法:
1. 使用 cuda 方法
2. 直接在 GPU 上创建张量
3. 使用 to 方法指定设备

```python
import torch


# 1. 使用 cuda 方法
def test01():

    data = torch.tensor([10, 20 ,30])
    print('存储设备:', data.device)

    # 如果安装的不是 gpu 版本的 PyTorch
    # 或电脑本身没有 NVIDIA 卡的计算环境
    # 下面代码可能会报错
    data = data.cuda()
    print('存储设备:', data.device)

    # 使用 cpu 函数将张量移动到 cpu 上
    data = data.cpu()
    print('存储设备:', data.device)

    # 输出结果:
    # 存储设备: cpu
    # 存储设备: cuda:0
    # 存储设备: cpu


# 2. 直接将张量创建在 GPU 上
def test02():

    data = torch.tensor([10, 20, 30], device='cuda:0')
    print('存储设备:', data.device)

    # 使用 cpu 函数将张量移动到 cpu 上
    data = data.cpu()
    print('存储设备:', data.device)

    # 输出结果:
    # 存储设备: cuda:0
    # 存储设备: cpu


# 3. 使用 to 方法
def test03():

    data = torch.tensor([10, 20, 30])
    print('存储设备:', data.device)

    data = data.to('cuda:0')
    print('存储设备:', data.device)

    # 输出结果:
    # 存储设备: cpu
    # 存储设备: cuda:0


# 4. 存储在不同设备的张量不能运算
def test04():

    data1 = torch.tensor([10, 20, 30], device='cuda:0')
    data2 = torch.tensor([10, 20, 30])
    print(data1.device, data2.device)

    # RuntimeError: Expected all tensors to be on the same device,
    # but found at least two devices, cuda:0 and cpu!
    data = data1 + data2
    print(data)


if __name__ == '__main__':
    test04()
```

程序输出结果:

```python
存储设备: cpu
存储设备: cuda:0
存储设备: cpu
存储设备: cuda:0
存储设备: cpu
存储设备: cpu
存储设备: cuda:0
cuda:0 cpu
```

---


# 3. 张量类型转换

掌握张量类型转换方法。


张量的类型转换也是经常使用的一种操作，是必须掌握的知识点。

## 张量转换为 numpy 数组

使用 Tensor.numpy 函数可以将张量转换为 ndarray 数组，但是共享内存，可以使用 copy 函数避免共享。

```python
# 1. 将张量转换为 numpy 数组
def test01():

    data_tensor = torch.tensor([2, 3, 4])
    # 使用张量对象中的 numpy 函数进行转换
    data_numpy = data_tensor.numpy()

    print(type(data_tensor))
    print(type(data_numpy))

    # 注意: data_tensor 和 data_numpy 共享内存
    # 修改其中的一个，另外一个也会发生改变
    # data_tensor[0] = 100
    data_numpy[0] = 100

    print(data_tensor)
    print(data_numpy)


if __name__ == '__main__':
    test01()
```

## numpy 转换为张量

1. 使用 from\_numpy 可以将 ndarray 数组转换为 Tensor，默认共享内存，使用 copy 函数避免共享。
2. 使用 torch.tensor 可以将 ndarray 数组转换为 Tensor，默认不共享内存。

```python
# 1. 使用 from_numpy 函数
def test01():

    data_numpy = np.array([2, 3, 4])
    # 将 numpy 数组转换为张量类型
    # 1. from_numpy
    # 2. torch.tensor(ndarray)

    # 浅拷贝
    data_tensor = torch.from_numpy(data_numpy)

    # nunpy 和 tensor 共享内存
    # data_numpy[0] = 100
    data_tensor[0] = 100

    print(data_tensor)
    print(data_numpy)


# 2. 使用 torch.tensor 函数
def test02():

    data_numpy = np.array([2, 3, 4])

    data_tensor = torch.tensor(data_numpy)

    # nunpy 和 tensor 不共享内存
    # data_numpy[0] = 100
    data_tensor[0] = 100

    print(data_tensor)
    print(data_numpy)


if __name__ == '__main__':
    test01()
    test02()
```

## 标量张量和数字的转换

对于只有一个元素的张量，使用 item 方法将该值从张量中提取出来。

```python
# 3. 标量张量和数字的转换
def test03():

    # 当张量只包含一个元素时, 可以通过 item 函数提取出该值
    data = torch.tensor([30,])
    print(data.item())

    data = torch.tensor(30)
    print(data.item())


if __name__ == '__main__':
    test03()
```

程序输出结果:

```python
30
30
```

---


# 4. 张量拼接操作

掌握 torch.cat 与 torch.stack 的使用。


张量的拼接操作在神经网络搭建过程中是非常常用的方法，例如: 在后面将要学习到的残差网络、注意力机制中都使用到了张量拼接。

## torch.cat 函数的使用

torch.cat 函数可以将两个张量根据指定的维度拼接起来.

```python
import torch


def test():

    data1 = torch.randint(0, 10, [3, 5, 4])
    data2 = torch.randint(0, 10, [3, 5, 4])

    print(data1)
    print(data2)
    print('-' * 50)

    # 1. 按0维度拼接
    new_data = torch.cat([data1, data2], dim=0)
    print(new_data.shape)
    print('-' * 50)

    # 2. 按1维度拼接
    new_data = torch.cat([data1, data2], dim=1)
    print(new_data.shape)
    print('-' * 50)

    # 3. 按2维度拼接
    new_data = torch.cat([data1, data2], dim=2)
    print(new_data.shape)


if __name__ == '__main__':
    test()
```

程序输出结果:

```python
tensor([[[6, 8, 3, 5],
         [1, 1, 3, 8],
         [9, 0, 4, 4],
         [1, 4, 7, 0],
         [5, 1, 4, 8]],

        [[0, 1, 4, 4],
         [4, 1, 8, 7],
         [5, 2, 6, 6],
         [2, 6, 1, 6],
         [0, 7, 8, 9]],

        [[0, 6, 8, 8],
         [5, 4, 5, 8],
         [3, 5, 5, 9],
         [3, 5, 2, 4],
         [3, 8, 1, 1]]])
tensor([[[4, 6, 8, 1],
         [0, 1, 8, 2],
         [4, 9, 9, 8],
         [5, 1, 5, 9],
         [9, 4, 3, 0]],

        [[7, 6, 3, 3],
         [4, 3, 3, 2],
         [2, 1, 1, 1],
         [3, 0, 8, 2],
         [8, 6, 6, 5]],

        [[0, 7, 2, 4],
         [4, 3, 8, 3],
         [4, 2, 1, 9],
         [4, 2, 8, 9],
         [3, 7, 0, 8]]])
--------------------------------------------------
torch.Size([6, 5, 4])
--------------------------------------------------
torch.Size([3, 10, 4])
tensor([[[6, 8, 3, 5, 4, 6, 8, 1],
         [1, 1, 3, 8, 0, 1, 8, 2],
         [9, 0, 4, 4, 4, 9, 9, 8],
         [1, 4, 7, 0, 5, 1, 5, 9],
         [5, 1, 4, 8, 9, 4, 3, 0]],

        [[0, 1, 4, 4, 7, 6, 3, 3],
         [4, 1, 8, 7, 4, 3, 3, 2],
         [5, 2, 6, 6, 2, 1, 1, 1],
         [2, 6, 1, 6, 3, 0, 8, 2],
         [0, 7, 8, 9, 8, 6, 6, 5]],

        [[0, 6, 8, 8, 0, 7, 2, 4],
         [5, 4, 5, 8, 4, 3, 8, 3],
         [3, 5, 5, 9, 4, 2, 1, 9],
         [3, 5, 2, 4, 4, 2, 8, 9],
         [3, 8, 1, 1, 3, 7, 0, 8]]])
```

## torch.stack 函数的使用

torch.stack 函数可以将两个张量根据指定的维度叠加起来.

```python
import torch


def test():

    data1= torch.randint(0, 10, [2, 3])
    data2= torch.randint(0, 10, [2, 3])
    print(data1)
    print(data2)

    new_data = torch.stack([data1, data2], dim=0)
    print(new_data.shape)

    new_data = torch.stack([data1, data2], dim=1)
    print(new_data.shape)

    new_data = torch.stack([data1, data2], dim=2)
    print(new_data.shape)


if __name__ == '__main__':
    test()
```

程序输出结果:

```python
tensor([[5, 8, 7],
        [6, 0, 6]])
tensor([[5, 8, 0],
        [9, 0, 1]])
torch.Size([2, 2, 3])
torch.Size([2, 2, 3])
torch.Size([2, 3, 2])
```

---


# 5. 张量索引操作

掌握张量不同索引操作。


我们在操作张量时，经常需要去进行获取或者修改操作，掌握张量的花式索引操作是必须的一项能力。

## 简单行、列索引

准备数据

```python
import torch

data = torch.randint(0, 10, [4, 5])
print(data)
print('-' * 50)
```

程序输出结果:

```python
tensor([[0, 7, 6, 5, 9],
        [6, 8, 3, 1, 0],
        [6, 3, 8, 7, 3],
        [4, 9, 5, 3, 1]])
--------------------------------------------------
```

```python
# 1. 简单行、列索引
def test01():

    print(data[0])
    print(data[:, 0])
    print('-' * 50)

if __name__ == '__main__':
    test01()
```

程序输出结果:

```python
tensor([0, 7, 6, 5, 9])
tensor([0, 6, 6, 4])
--------------------------------------------------
```

## 列表索引

```python
# 2. 列表索引
def test02():

    # 返回 (0, 1)、(1, 2) 两个位置的元素
    print(data[[0, 1], [1, 2]])
    print('-' * 50)

    # 返回 0、1 行的 1、2 列共4个元素
    print(data[[[0], [1]], [1, 2]])
if __name__ == '__main__':
    test02()
```

程序输出结果:

```python
tensor([7, 3])
--------------------------------------------------
tensor([[7, 6],
        [8, 3]])
```

## 范围索引

```python
# 3. 范围索引
def test03():
    # 前3行的前2列数据
    print(data[:3, :2])
    # 第2行到最后的前2列数据
    print(data[2:, :2])
if __name__ == '__main__':
    test03()
```

程序输出结果:

```python
tensor([[0, 7],
        [6, 8],
        [6, 3]])
tensor([[6, 3],
        [4, 9]])
```

## 布尔索引

```python
# 布尔索引
def test():

    # 第2列大于5的行数据
    print(data[data[:, 2] > 5])
    # 第1行大于5的列数据
    print(data[:, data[1] > 5])
if __name__ == '__main__':
    test04()
```

程序输出结果:

```python
tensor([[0, 7, 6, 5, 9],
        [6, 3, 8, 7, 3]])
tensor([[0, 7],
        [6, 8],
        [6, 3],
        [4, 9]])
```

## 多维索引

```python
# 多维索引
def test05():

    data = torch.randint(0, 10, [3, 4, 5])
    print(data)
    print('-' * 50)

    print(data[0, :, :])
    print(data[:, 0, :])
    print(data[:, :, 0])


if __name__ == '__main__':
    test05()
```

程序输出结果:

```python
tensor([[[2, 4, 1, 2, 3],
         [5, 5, 1, 5, 0],
         [1, 4, 5, 3, 8],
         [7, 1, 1, 9, 9]],

        [[9, 7, 5, 3, 1],
         [8, 8, 6, 0, 1],
         [6, 9, 0, 2, 1],
         [9, 7, 0, 4, 0]],

        [[0, 7, 3, 5, 6],
         [2, 4, 6, 4, 3],
         [2, 0, 3, 7, 9],
         [9, 6, 4, 4, 4]]])
--------------------------------------------------
tensor([[2, 4, 1, 2, 3],
        [5, 5, 1, 5, 0],
        [1, 4, 5, 3, 8],
        [7, 1, 1, 9, 9]])
tensor([[2, 4, 1, 2, 3],
        [9, 7, 5, 3, 1],
        [0, 7, 3, 5, 6]])
tensor([[2, 5, 1, 7],
        [9, 8, 6, 9],
        [0, 2, 2, 9]])
```

---


# 6. 张量形状操作

掌握 reshape、transpose、permute、view、contiguous、squeeze、unsqueeze 等函数的使用。


在我们后面搭建网络模型时，数据都是基于张量形式的表示，网络层与层之间很多都是以不同的 shape 的方式进行表现和运算，我们需要掌握对张量形状的操作，以便能够更好处理网络各层之间的数据连接。

## reshape 函数的用法

reshape 函数可以在保证张量数据不变的前提下改变数据的维度，将其转换成指定的形状，在后面的神经网络学习时，会经常使用该函数来调节数据的形状，以适配不同网络层之间的数据传递。

```python
import torch
import numpy as np


def test():

    data = torch.tensor([[10, 20, 30], [40, 50, 60]])

    # 1. 使用 shape 属性或者 size 方法都可以获得张量的形状
    print(data.shape, data.shape[0], data.shape[1])
    print(data.size(), data.size(0), data.size(1))

    # 2. 使用 reshape 函数修改张量形状
    new_data = data.reshape(1, 6)
    print(new_data.shape)


if __name__ == '__main__':
    test()
```

程序运行结果:

```python
torch.Size([2, 3]) 2 3
torch.Size([2, 3]) 2 3
torch.Size([1, 6])
```

## transpose 和 permute 函数的使用

transpose 函数可以实现交换张量形状的指定维度, 例如: 一个张量的形状为 (2, 3, 4) 可以通过 transpose 函数把 3 和 4 进行交换, 将张量的形状变为 (2, 4, 3)

permute 函数可以一次交换更多的维度。

```python
import torch
import numpy as np


def test():

    data = torch.tensor(np.random.randint(0, 10, [3, 4, 5]))
    print('data shape:', data.size())

    # 1. 交换1和2维度
    new_data = torch.transpose(data, 1, 2)
    print('data shape:', new_data.size())

    # 2. 将 data 的形状修改为 (4, 5, 3)
    new_data = torch.transpose(data, 0, 1)
    new_data = torch.transpose(new_data, 1, 2)
    print('new_data shape:', new_data.size())

    # 3. 使用 permute 函数将形状修改为 (4, 5, 3)
    new_data = torch.permute(data, [1, 2, 0])
    print('new_data shape:', new_data.size())


if __name__ == '__main__':
    test()
```

程序运行结果:

```python
data shape: torch.Size([3, 4, 5])
data shape: torch.Size([3, 5, 4])
new_data shape: torch.Size([4, 5, 3])
new_data shape: torch.Size([4, 5, 3])
```

## view 和 contigous 函数的用法

view 函数也可以用于修改张量的形状，但是其用法比较局限，只能用于存储在整块内存中的张量。在 PyTorch 中，有些张量是由不同的数据块组成的，它们并没有存储在整块的内存中，view 函数无法对这样的张量进行变形处理，例如: 一个张量经过了 transpose 或者 permute 函数的处理之后，就无法使用 view 函数进行形状操作。

```python
import torch
import numpy as np


def test():

    data = torch.tensor([[10, 20, 30], [40, 50, 60]])
    print('data shape:', data.size())

    # 1. 使用 view 函数修改形状
    new_data = data.view(3, 2)
    print('new_data shape:', new_data.shape)

    # 2. 判断张量是否使用整块内存
    print('data:', data.is_contiguous())  # True

    # 3. 使用 transpose 函数修改形状
    new_data = torch.transpose(data, 0, 1)
    print('new_data:', new_data.is_contiguous())  # False
    # new_data = new_data.view(2, 3)  # RuntimeError

    # 需要先使用 contiguous 函数转换为整块内存的张量，再使用 view 函数
    print(new_data.contiguous().is_contiguous())
    new_data = new_data.contiguous().view(2, 3)
    print('new_data shape:', new_data.shape)


if __name__ == '__main__':
    test()
```

程序运行结果:

```python
data shape: torch.Size([2, 3])
new_data shape: torch.Size([3, 2])
data: True
new_data: False
True
new_data shape: torch.Size([2, 3])
```

## squeeze 和 unsqueeze 函数的用法

squeeze 函数用删除 shape 为 1 的维度，unsqueeze 在每个维度添加 1, 以增加数据的形状。

```python
import torch
import numpy as np


def test():

    data = torch.tensor(np.random.randint(0, 10, [1, 3, 1, 5]))
    print('data shape:', data.size())

    # 1. 去掉值为1的维度
    new_data = data.squeeze()
    print('new_data shape:', new_data.size())  # torch.Size([3, 5])

    # 2. 去掉指定位置为1的维度，注意: 如果指定位置不是1则不删除
    new_data = data.squeeze(2)
    print('new_data shape:', new_data.size())  # torch.Size([3, 5])

    # 3. 在2维度增加一个维度
    new_data = data.unsqueeze(-1)
    print('new_data shape:', new_data.size())  # torch.Size([3, 1, 5, 1])


if __name__ == '__main__':
    test()
```

程序运行结果:

```python
data shape: torch.Size([1, 3, 1, 5])
new_data shape: torch.Size([3, 5])
new_data shape: torch.Size([1, 3, 5])
new_data shape: torch.Size([1, 3, 1, 5, 1])
```

---


# 7. 张量运算函数

掌握张量相关运算函数。


## 常见运算函数

PyTorch 为每个张量封装很多实用的计算函数，例如计算均值、平方根、求和等等

```python
import torch


def test():

    data = torch.randint(0, 10, [2, 3], dtype=torch.float64)
    print(data)
    print('-' * 50)

    # 1. 计算均值
    # 注意: tensor 必须为 Float 或者 Double 类型
    print(data.mean())
    print(data.mean(dim=0))  # 按列计算均值
    print(data.mean(dim=1))  # 按行计算均值
    print('-' * 50)

    # 2. 计算总和
    print(data.sum())
    print(data.sum(dim=0))
    print(data.sum(dim=1))
    print('-' * 50)

    # 3. 计算平方
    print(data.pow(2))
    print('-' * 50)

    # 4. 计算平方根
    print(data.sqrt())
    print('-' * 50)

    # 5. 指数计算, e^n 次方
    print(data.exp())
    print('-' * 50)

    # 6. 对数计算
    print(data.log())  # 以 e 为底
    print(data.log2())
    print(data.log10())


if __name__ == '__main__':
    test()
```

程序运行结果:

```python
tensor([[4., 0., 7.],
        [6., 3., 5.]], dtype=torch.float64)
--------------------------------------------------
tensor(4.1667, dtype=torch.float64)
tensor([5.0000, 1.5000, 6.0000], dtype=torch.float64)
tensor([3.6667, 4.6667], dtype=torch.float64)
--------------------------------------------------
tensor(25., dtype=torch.float64)
tensor([10.,  3., 12.], dtype=torch.float64)
tensor([11., 14.], dtype=torch.float64)
--------------------------------------------------
tensor([[16.,  0., 49.],
        [36.,  9., 25.]], dtype=torch.float64)
--------------------------------------------------
tensor([[2.0000, 0.0000, 2.6458],
        [2.4495, 1.7321, 2.2361]], dtype=torch.float64)
--------------------------------------------------
tensor([[5.4598e+01, 1.0000e+00, 1.0966e+03],
        [4.0343e+02, 2.0086e+01, 1.4841e+02]], dtype=torch.float64)
--------------------------------------------------
tensor([[1.3863,   -inf, 1.9459],
        [1.7918, 1.0986, 1.6094]], dtype=torch.float64)
tensor([[2.0000,   -inf, 2.8074],
        [2.5850, 1.5850, 2.3219]], dtype=torch.float64)
tensor([[0.6021,   -inf, 0.8451],
        [0.7782, 0.4771, 0.6990]], dtype=torch.float64)
```

---


# 8. 自动微分模块

掌握梯度计算。


自动微分（Autograd）模块对张量做了进一步的封装，具有自动求导功能。自动微分模块是构成神经网络训练的必要模块，在神经网络的反向传播过程中，Autograd 模块基于正向计算的结果对当前的参数进行微分计算，从而实现网络权重参数的更新。

## 梯度基本计算

我们使用 backward 方法、grad 属性来实现梯度的计算和访问.

```python
import torch


# 1. 单标量梯度的计算
# y = x**2 + 20
def test01():

    # 定义需要求导的张量
    # 张量的值类型必须是浮点类型
    x = torch.tensor(10, requires_grad=True, dtype=torch.float64)
    # 变量经过中间运算
    f = x ** 2 + 20
    # 自动微分
    f.backward()
    # 打印 x 变量的梯度
    # backward 函数计算的梯度值会存储在张量的 grad 变量中
    print(x.grad)


# 2. 单向量梯度的计算
# y = x**2 + 20
def test02():

    # 定义需要求导张量
    x = torch.tensor([10, 20, 30, 40], requires_grad=True, dtype=torch.float64)
    # 变量经过中间计算
    f1 = x ** 2 + 20

    # 注意:
    # 由于求导的结果必须是标量
    # 而 f 的结果是: tensor([120., 420.])
    # 所以, 不能直接自动微分
    # 需要将结果计算为标量才能进行计算
    f2 = f1.mean()  # f2 = 1/2 * x

    # 自动微分
    f2.backward()

    # 打印 x 变量的梯度
    print(x.grad)


if __name__ == '__main__':
    test01()
```

程序运行结果:

```python
tensor(20., dtype=torch.float64)
tensor([ 5., 10., 15., 20.], dtype=torch.float64)
```

## 控制梯度计算

我们可以通过一些方法使得在 requires\_grad=True 的张量在某些时候计算不进行梯度计算。

```python
import torch


# 1. 控制不计算梯度
def test01():

    x = torch.tensor(10, requires_grad=True, dtype=torch.float64)
    print(x.requires_grad)

    # 第一种方式: 对代码进行装饰
    with torch.no_grad():
        y = x ** 2
        print(y.requires_grad)

    # 第二种方式: 对函数进行装饰
    @torch.no_grad()
    def my_func(x):
        return x ** 2
    print(my_func(x).requires_grad)


    # 第三种方式
    torch.set_grad_enabled(False)
    y = x ** 2
    print(y.requires_grad)


# 2. 注意: 累计梯度
def test02():

    # 定义需要求导张量
    x = torch.tensor([10, 20, 30, 40], requires_grad=True, dtype=torch.float64)

    for _ in range(3):

        f1 = x ** 2 + 20
        f2 = f1.mean()

        # 默认张量的 grad 属性会累计历史梯度值
        # 所以, 需要我们每次手动清理上次的梯度
        # 注意: 一开始梯度不存在, 需要做判断
        if x.grad is not None:
            x.grad.data.zero_()

        f2.backward()
        print(x.grad)


# 3. 梯度下降优化最优解
def test03():

    # y = x**2
    x = torch.tensor(10, requires_grad=True, dtype=torch.float64)

    for _ in range(5000):

        # 正向计算
        f = x ** 2

        # 梯度清零
        if x.grad is not None:
            x.grad.data.zero_()

        # 反向传播计算梯度
        f.backward()

        # 更新参数
        x.data = x.data - 0.001 * x.grad

        print('%.10f' % x.data)


if __name__ == '__main__':
    test01()
    print('--------------------')
    test02()
    print('--------------------')
    test03()
```

程序运行结果:

```python
True
False
False
False
tensor([ 5., 10., 15., 20.], dtype=torch.float64)
tensor([ 5., 10., 15., 20.], dtype=torch.float64)
tensor([ 5., 10., 15., 20.], dtype=torch.float64)
```

## 梯度计算注意

当对设置 requires\_grad=True 的张量使用 numpy 函数进行转换时, 会出现如下报错:

> Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.

此时, 需要先使用 detach 函数将张量进行分离, 再使用 numpy 函数.

注意: detach 之后会产生一个新的张量, 新的张量作为叶子结点，并且该张量和原来的张量共享数据, 但是分离后的张量不需要计算梯度。

```python
import torch


# 1. detach 函数用法
def test01():

    x = torch.tensor([10, 20], requires_grad=True, dtype=torch.float64)

    # Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    # print(x.numpy())  # 错误
    print(x.detach().numpy())  # 正确


# 2. detach 前后张量共享内存
def test02():

    x1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float64)

    # x2 作为叶子结点
    x2 = x1.detach()

    # 两个张量的值一样: 140421811165776 140421811165776
    print(id(x1.data), id(x2.data))
    x2.data = torch.tensor([100, 200])
    print(x1)
    print(x2)

    # x2 不会自动计算梯度: False
    print(x2.requires_grad)


if __name__ == '__main__':
    test01()
    test02()
```

程序运行结果:

```python
10. 20.]
140495634222288 140495634222288
tensor([10., 20.], dtype=torch.float64, requires_grad=True)
tensor([100, 200])
False
```

---


# 9. 模型的保存与加载

掌握 PyTorch 保存模型的方法。


- 神经网络的训练有时需要几天, 几周, 甚至几个月, 为了在每次使用模型时避免高代价的重复训练, 我们就需要将模型序列化到磁盘中, 使用的时候反序列化到内存中.

#### 保存模型参数

```python
import torch
import torch.nn as nn


# 假设我们有一个模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()

# 保存模型的参数
torch.save(model.state_dict(), 'model_weights.pth')
```

#### 保存全部模型

```python
import torch
import torch.nn as nn


# 假设我们有一个模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()

# 保存全部模型
torch.save(model, 'model.pth')
```

#### 加载模型参数

```python
# 创建一个与保存时相同结构的模型
model = SimpleModel()

# 加载模型的参数
model.load_state_dict(torch.load('model_weights.pth'))
print(model)
```

#### 加载全部模型

```python
model = torch.load('model.pth')
print(model)
```

- 注意📢:
- 模型结构: 如果你只保存了模型的参数, 那么在加载时需要确保你有与保存时相同的模型结构.
- 设备兼容性: 如果你在一个设备上保存了模型(例如GPU), 而在另一个设备上加载(例如CPU), 你可能需要使用 `map_location` 参数来指定设备.

```python
device = torch.device('cpu')
    model.load_state_dict(torch.load('model_weights.pth', map_location=device))
```

---

# 10. torch.nn 神经网络模块

> 前面的章节介绍的都是「张量」本身的运算。从本章开始，进入**神经网络模块**（`torch.nn`）体系——
> 它是用 PyTorch 搭模型的核心。本章讲解模块组织、常用层类型与参数管理。

## nn.Module：模型组织基础

在 PyTorch 中，任何模型/层都继承自 `nn.Module`。它负责三件事：
**组织子模块**、**自动管理参数**、**提供设备迁移与序列化**。

- 把子模块（`nn.Linear`、`nn.LayerNorm` 等）挂成 `self.xxx` 属性，它们会自动被注册
- `named_parameters()` 遍历全部参数（含嵌套子模块），`state_dict()` 导出参数做保存/加载
- 对模型调用 `.to(device)` 会把所有参数与 buffer 一起迁移

示例：

```python
import torch
import torch.nn as nn


class MyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)      # 子模块自动注册
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.fc(x))


m = MyBlock(16)
print('子模块:', list(m.children()))
print('参数名:', [n for n, _ in m.named_parameters()])
print('state_dict keys:', list(m.state_dict().keys()))
```

程序输出结果:

```python
子模块: [Linear(in_features=16, out_features=16, bias=True), LayerNorm((16,), eps=1e-05, elementwise_affine=True, bias=True)]
参数名: ['fc.weight', 'fc.bias', 'norm.weight', 'norm.bias']
state_dict keys: ['fc.weight', 'fc.bias', 'norm.weight', 'norm.bias']
```

要点：

- `forward()` 定义前向计算逻辑，**不要直接调用它**，应通过 `m(x)` 触发
- 参数名带前缀（如 `fc.weight`），嵌套模块越多前缀越长，便于定位
- `state_dict()` 的结果可直接交给 `torch.save` / `load_state_dict`（与第 9 章配合使用）

## nn.Linear：全连接层

`nn.Linear(in_features, out_features, bias=True)` 实现 `y = xW^T + b`，
是最常用的层之一（MLP、分类头、维度投影都用它）。

- 权重形状为 `[out_features, in_features]`，偏置形状为 `[out_features]`
- `bias=False` 时不创建偏置（此时 `bias` 为 `None`）
- 输入 `[..., in_features]`，输出 `[..., out_features]`，前面的维度不变

示例：

```python
import torch
import torch.nn as nn

fc = nn.Linear(4, 8, bias=True)
x = torch.randn(2, 4)
print('输入 [2,4] -> 输出:', fc(x).shape)
print('权重形状:', fc.weight.shape, '| 偏置形状:', fc.bias.shape)

fc_nb = nn.Linear(4, 8, bias=False)
print('bias=False 时 bias 为:', fc_nb.bias)
```

程序输出结果:

```python
输入 [2,4] -> 输出: torch.Size([2, 8])
权重形状: torch.Size([8, 4]) | 偏置形状: torch.Size([8])
bias=False 时 bias 为: None
```

## nn.Embedding：查表嵌入

`nn.Embedding(num_embeddings, embedding_dim)` 维护一张可学习的
`[num_embeddings, embedding_dim]` 查找表，输入 token id（整型）直接查出对应行。
它是 NLP / 多模态模型把离散 token 变成稠密向量的标准做法。

- 输入形状 `[N, ...]`（任意整型），输出比输入多一维 `embedding_dim`
- `weight` 就是查找表本身，随训练更新

示例：

```python
import torch
import torch.nn as nn

emb = nn.Embedding(10, 4)            # 词表 10，向量维 4
idx = torch.tensor([1, 3, 1])        # 3 个 token id
print('查表 [3] ->', emb(idx).shape)
print('权重表形状:', emb.weight.shape)
```

程序输出结果:

```python
查表 [3] -> torch.Size([3, 4])
权重表形状: torch.Size([10, 4])
```

注意：Embedding 的输入必须是**整型**（`torch.long`），传入浮点会报错。

## nn.LayerNorm：层归一化

`nn.LayerNorm(normalized_shape)` 对最后一维做归一化：
把每个样本的特征向量调整为「均值 0、标准差 1」，再乘可学习的缩放 `weight`
并加可学习的偏移 `bias`。它不依赖 batch 内其他样本，因此常用于
Transformer、RNN 以及 batch 很小时。

示例：

```python
import torch
import torch.nn as nn

ln = nn.LayerNorm(4)
x = torch.randn(2, 4)
y = ln(x)
print('LayerNorm 输出形状:', y.shape)
print('每行均值≈0:', y.mean(dim=1))
print('每行标准差≈1:', y.std(dim=1, unbiased=False))
```

程序输出结果:

```python
LayerNorm 输出形状: torch.Size([2, 4])
每行均值≈0: tensor([-2.6077e-08,  2.9802e-08], grad_fn=<MeanBackward1>) 每行标准差≈1: tensor([1.0000, 1.0000], grad_fn=<StdBackward0>)
```

## nn.Conv2d：二维卷积

`nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)` 是
图像/音频等网格数据的标准层。输出尺寸遵循公式：
`out = floor((H + 2*padding - kernel) / stride + 1)`。

以 `kernel=3, stride=2, padding=1` 为例，`80 -> 40`、`20 -> 10`，恰好**长度减半**，
因此常被串联使用做逐级下采样。

示例：

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(1, 8, 3, 2, padding=1)   # 输入 1 通道 -> 输出 8 通道
x = torch.randn(2, 1, 80, 20)             # [batch, 通道, 高, 宽]
print('Conv2d(1->8, k3 s2 p1) 输入[2,1,80,20] ->', conv(x).shape)
```

程序输出结果:

```python
Conv2d(1->8, k3 s2 p1) 输入[2,1,80,20] -> torch.Size([2, 8, 40, 10])
```

## nn.ModuleList 与 nn.Sequential：容器

- `nn.ModuleList`：存放同构模块的列表，保证里面的参数被自动注册（普通 Python list 不会）
- `nn.Sequential`：把多个层**顺序连接**成一个整体，直接 `seq(x)` 依次前向

示例：

```python
import torch
import torch.nn as nn

# ModuleList：常用于「按层数循环构建」的编码器
ml = nn.ModuleList([nn.Linear(16, 16) for _ in range(3)])
print('ModuleList 长度:', len(ml))

# Sequential：顺序堆叠的小网络
seq = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
print('Sequential 前向:', seq(torch.randn(3, 4)).shape)
```

程序输出结果:

```python
ModuleList 长度: 3
Sequential 前向: torch.Size([3, 2])
```

## register_buffer：非参数缓冲

`register_buffer(name, tensor, persistent=True)` 把一个张量注册进模块，
它具备三个特性：

- **不是参数**：不进优化器、不计算梯度
- **随模块迁移**：`.to(device)` 时 buffer 一起走
- `persistent=True` 时进入 `state_dict()`（随 checkpoint 保存），`False` 则不保存

常用来存放**常量**（位置偏移、缩放系数等），例如为多组码本准备偏移量：

示例：

```python
import torch
import torch.nn as nn


class Demo(nn.Module):
    def __init__(self):
        super().__init__()
        # persistent=False：只在内存中使用，不写进 checkpoint
        self.register_buffer('offset', torch.arange(8), persistent=False)


d = Demo()
print('buffer:', d.offset, '| state_dict:', list(d.state_dict().keys()))
print('named_buffers:', [n for n, _ in d.named_buffers()])
```

程序输出结果:

```python
buffer: tensor([0, 1, 2, 3, 4, 5, 6, 7]) | state_dict: []
named_buffers: ['offset']
```

## 模块的设备与参数查询

- `next(model.parameters()).device`：获取模块所在设备（模块可能没有参数，此时应查 buffer）
- `sum(p.numel() for p in model.parameters())`：统计参数总数

示例：

```python
import torch
import torch.nn as nn

model = nn.Linear(4, 4)
print('next(parameters()).device:', next(model.parameters()).device)
print('参数总数:', sum(p.numel() for p in model.parameters()))
```

程序输出结果:

```python
next(parameters()).device: cpu
参数总数: 20
```

---

# 11. torch.nn.functional 函数库

## functional 与 nn 模块的关系

`torch.nn.functional`（惯例命名为 `F`）提供**无状态**的数学函数；
同名模块（如 `nn.ReLU`）只是把这些函数包装成了带参数/可挂载的类。
选择原则：

| 场景 | 推荐 |
|---|---|
| 网络层（有可学习参数） | `nn.Linear`、`nn.Conv2d`、`nn.Embedding` |
| 无参数的操作，写在 `forward` 里 | `F.relu`、`F.gelu`、`F.pad`、`F.softmax` |
| 训练循环中的损失 | `F.cross_entropy`、`F.mse_loss` |

模块形式的好处是能放进 `nn.Sequential`、被 `state_dict()` 管理；
函数形式的好处是轻量、不需要额外对象。

## 激活函数：relu / silu / gelu

三种常见激活函数对同一输入的输出对比（输入 `[-2, -1, 0, 1, 2]`）：

- `F.relu`：负数一律为 0（最简单、最常用）
- `F.silu`：`x * sigmoid(x)`，负区间有轻微负值（平滑）
- `F.gelu`：高斯误差线性单元，Transformer 系模型常用（平滑）

示例：

```python
import torch
import torch.nn.functional as F

z = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print('F.relu:', F.relu(z))
print('F.silu:', F.silu(z))
print('F.gelu:', F.gelu(z))
```

程序输出结果:

```python
F.relu: tensor([0., 0., 0., 1., 2.])
F.silu: tensor([-0.2384, -0.2689,  0.0000,  0.7311,  1.7616])
F.gelu: tensor([-0.0455, -0.1587,  0.0000,  0.8413,  1.9545])
```

## F.pad：填充

`F.pad(x, pad, value)` 按 `(左, 右, 上, 下, ...)` 顺序填充张量边缘。
一维场景下 `(left, right)` 对应首尾；也常用「头部补一个值」来构造前缀边界。

示例：

```python
import torch
import torch.nn.functional as F

x = torch.tensor([[1, 2, 3]])
print('尾部补2个0:', F.pad(x, (0, 2), value=0))
print('左1右1补-1:', F.pad(x, (1, 1), value=-1))

cs = torch.tensor([12, 8], dtype=torch.int32)
print('头部补0 (cu_seqlens):', F.pad(cs, (1, 0), value=0))
```

程序输出结果:

```python
尾部补2个0: tensor([[1, 2, 3, 0, 0]])
左1右1补-1: tensor([[-1,  1,  2,  3, -1]])
头部补0 (cu_seqlens): tensor([ 0, 12,  8], dtype=torch.int32)
```

## 其他常用 F 函数

- `F.softmax(x, dim)`：把 logits 归一化成概率，**行和为 1**
- `F.cross_entropy(logits, target)`：交叉熵损失，内部已含 softmax
  （传入 logits 而非概率，数值上更稳定）
- 还有 `F.sigmoid` / `F.tanh` / `F.dropout` / `F.mse_loss` 等，用法类似

示例：

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1]])
print('softmax:', F.softmax(logits, dim=-1))
print('softmax 行和:', F.softmax(logits, dim=-1).sum(dim=-1))

target = torch.tensor([0])                       # 真实类别
print('cross_entropy:', F.cross_entropy(logits, target))
```

程序输出结果:

```python
softmax: tensor([[0.6590, 0.2424, 0.0986]])
softmax 行和: tensor([1.0000])
cross_entropy: tensor(0.4170)
```

---

# 12. 设备、数据类型与内存管理

## 设备与 .device

张量可以存放在 CPU 或 GPU（CUDA）上，通过 `.device` 查看。
**参与同一运算的所有张量必须在同一设备、同一 dtype 上**，否则报错。

- 创建时指定：`torch.zeros(3, device='cuda:0')`
- 运行时迁移：`x.to('cuda:0')` / `x.to(device)`
- 查询：`x.device`、`torch.cuda.is_available()`

实际工程中通常先写 `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`，
再统一 `.to(device)`。

## 数据类型体系

PyTorch 常用 dtype：

| dtype | 说明 | 典型用途 |
|---|---|---|
| `torch.bool` | 布尔 | 掩码、条件判断 |
| `torch.int32` | 32 位整数 | 边界索引、计数 |
| `torch.int64`（`long`） | 64 位整数 | **token id、下标**（默认整型） |
| `torch.float32` | 单精度浮点 | 默认浮点精度 |
| `torch.float16` | 半精度浮点 | GPU 显存优化（范围小） |
| `torch.bfloat16` | 脑浮点 | 大模型训练/推理的主流低精度 |

示例：

```python
import torch

print('bool:', torch.zeros(3, dtype=torch.bool).dtype)
print('int32:', torch.zeros(3, dtype=torch.int32).dtype)
print('int64(long):', torch.zeros(3, dtype=torch.long).dtype)
print('float32:', torch.zeros(3, dtype=torch.float32).dtype)
print('float16:', torch.zeros(3, dtype=torch.float16).dtype)
print('bfloat16:', torch.zeros(3, dtype=torch.bfloat16).dtype)
```

程序输出结果:

```python
bool: torch.bool
int32: torch.int32
int64(long): torch.int64
float32: torch.float32
float16: torch.float16
bfloat16: torch.bfloat16
```

## bfloat16：低精度但范围大

`bfloat16` 用 8 位指数 + 7 位尾数（与 float32 的指数位相同），
因此**数值范围与 float32 一致**、精度略低，比 float16 更抗溢出。
大模型推理常用它把权重和激活降到 bfloat16，以省一半显存并加速。

示例（精度与范围的直观对比）：

```python
import torch

a = torch.tensor(0.1, dtype=torch.float32)
b = torch.tensor(0.1, dtype=torch.bfloat16)
print('float32 表示 0.1:', a.item())
print('bfloat16 表示 0.1:', b.item())
print('float32 大数:', torch.tensor(1e5, dtype=torch.float32).item())
print('bfloat16 大数:', torch.tensor(1e5, dtype=torch.bfloat16).item())
```

程序输出结果:

```python
float32 表示 0.1: 0.10000000149011612
bfloat16 表示 0.1: 0.10009765625
float32 大数: 100000.0
bfloat16 大数: 99840.0
```

可以看到：bfloat16 对 0.1 的表示有轻微误差（精度降低），但 1e5 级别的大数
依然能表示（float16 此时早已溢出为 inf）。

## .to(device, dtype)：统一迁移

`.to()` 可同时指定 `device` 与 `dtype`，是模型代码中最常用的迁移手段。
常见写法：

- `x.to(device)`：迁移设备
- `x.to(dtype=torch.bfloat16)`：转精度
- `x.to(device=device, dtype=torch.bfloat16)`：两者一起

示例：

```python
import torch

x = torch.tensor([10, 20, 30])
y = x.to(dtype=torch.bfloat16)
print('dtype:', x.dtype, '->', y.dtype, '| device:', x.device)
```

程序输出结果:

```python
dtype: torch.int64 -> torch.bfloat16 | device: cpu
```

## torch.as_tensor：数据转换

`torch.as_tensor` 把 Python 列表 / numpy 数组 / 已有张量转为张量，
**尽量共享底层内存（不做拷贝）**。它与第 3 章讲的 `torch.from_numpy` 的区别：

- `torch.from_numpy` 只能接收 numpy 数组
- `torch.as_tensor` 更通用，且 `dtype` 不一致时自动转换（共享内存仅在类型兼容时成立）

示例：

```python
import torch
import numpy as np

a = torch.as_tensor([1, 2, 3], dtype=torch.long)
print('as_tensor:', a, a.dtype)

arr = np.array([[4.0, 5.0]])
b = torch.as_tensor(arr)
print('as_tensor(numpy):', b, b.dtype)
```

程序输出结果:

```python
as_tensor: tensor([1, 2, 3]) torch.int64
as_tensor(numpy): tensor([[4., 5.]], dtype=torch.float64) torch.float64
```

## 就地操作：clone / copy_ / zero_

- `clone()`：**深拷贝**，返回独立内存的新张量，用于保存不被后续覆盖的快照
- `copy_(src)`：把 src 数据**就地**拷入目标张量（下划线后缀 = 原地修改）
- `zero_()`：就地清零

就地操作（带下划线）直接修改原张量、不分配新内存，适合复用缓冲区、
省去每步的分配开销。

示例：

```python
import torch

x = torch.randn(2, 3)
c = x.clone()
print('clone 相等:', torch.equal(x, c))

buf = torch.zeros(2, 3)
buf.copy_(x)
print('copy_ 相等:', torch.equal(buf, x))
buf.zero_()
print('zero_ 后:', buf.sum().item())
```

程序输出结果:

```python
clone 相等: True
copy_ 相等: True
zero_ 后: 0.0
```

## 数值裁剪：clamp

`clamp(min, max)` 把数值裁剪进 `[min, max]` 区间，常用于：
限制音频波形振幅、裁剪梯度、约束输出范围。

示例：

```python
import torch

wav = torch.tensor([-2.5, -1.0, 0.3, 1.0, 3.2])
print('clamp[-1,1]:', wav.clamp(min=-1.0, max=1.0))
print('clamp(min=0):', torch.tensor([-1.0, 0.5, 2.0]).clamp(min=0))
```

程序输出结果:

```python
clamp[-1,1]: tensor([-1.0000, -1.0000,  0.3000,  1.0000,  1.0000])
clamp(min=0): tensor([0.0000, 0.5000, 2.0000])
```

## 显存估算：numel / element_size

- `numel()`：元素总数
- `element_size()`：每个元素占用的字节数

两者相乘即可估算张量/模型参数占用的显存。加载模型后常用它打印
「Model Loaded」日志（例如 1 个 bfloat16 的 4×8 参数 = 32 元素 × 2 字节 = 64 字节）。

示例：

```python
import torch

p = torch.randn(4, 8)
print('numel:', p.numel(), 'element_size:', p.element_size(),
      'bytes:', p.numel() * p.element_size())
```

程序输出结果:

```python
numel: 32 element_size: 4 bytes: 128
```

## 高效扩展：expand 与 repeat

两者都能把一个张量沿维度变大，但机制不同：

- `expand`：把**大小为 1 的维度**扩展，**零拷贝**（返回视图），`-1` 表示保持该维度
- `repeat`：真正**复制数据**（返回新张量），更灵活但更占内存

批量场景中常用 `expand` 把一个 `[1, H]` 的向量复制成 `[N, H]`，避免浪费内存。

示例：

```python
import torch

pad = torch.randn(1, 2048)
print('expand(4,-1):', pad.expand(4, -1).shape, '| 共享内存视图')

z = torch.zeros_like(torch.tensor([10, 20, 30])).reshape(-1, 1)
print('repeat(1,8):', z.repeat(1, 8).shape)

e = torch.randn(1, 3)
print('expand 值:', e.expand(4, 3))     # 4 行完全相同
```

程序输出结果:

```python
expand(4,-1): torch.Size([4, 2048]) | 共享内存视图
repeat(1,8): torch.Size([3, 8])
expand 值: tensor([[ 2.3139, -0.8249,  0.2091],
        [ 2.3139, -0.8249,  0.2091],
        [ 2.3139, -0.8249,  0.2091],
        [ 2.3139, -0.8249,  0.2091]])
```

---

# 13. 进阶实用技巧

> 本章汇总前面各章没有单独讲到的实用 API 与**组合模式**：
> 推理模式、变长序列处理、注意力边界构建、掩码抑制、组合查表等。

## 推理模式：inference_mode 与 no_grad

`torch.no_grad()` 和 `torch.inference_mode()` 都用于**关闭梯度追踪**，
区别在于：

- `no_grad()`：不记录梯度，但张量仍保持「可训练」语义（例如仍可被要求 requires_grad）
- `inference_mode()`：更激进，进一步禁用自动微分相关机制，**推理更快**；
  只用于纯前向、不需要任何梯度的场景

推理阶段（生成、评估）应该用 `inference_mode()` 包裹前向计算。

示例：

```python
import torch

with torch.inference_mode():
    y = torch.randn(2, 3) * 2
    print('inference_mode 内:', y.shape, '| requires_grad:', y.requires_grad)
```

程序输出结果:

```python
inference_mode 内: torch.Size([2, 3]) | requires_grad: False
```

## 编译追踪与默认精度：jit.is_tracing / get_default_dtype

- `torch.jit.is_tracing()`：判断当前是否处于 TorchScript / `torch.compile` 的
  追踪（tracing）阶段。图编译要求数据类型一致，因此代码里常用它做
  「追踪时沿用输入 dtype、否则用固定 dtype」的分支
- `torch.get_default_dtype()`：返回默认浮点精度（通常 `torch.float32`），
  需要查询全局默认精度时使用

示例：

```python
import torch

print('is_tracing:', torch.jit.is_tracing())
print('get_default_dtype:', torch.get_default_dtype())
```

程序输出结果:

```python
is_tracing: False
get_default_dtype: torch.float32
```

## 变长序列处理：split / chunk / pad_sequence

- `torch.split(tensor, sizes_or_count, dim)`：按**长度列表**或**等长块数**切分
- `tensor.chunk(n, dim)`：均分为 **n 份**
- `nn.utils.rnn.pad_sequence(list, batch_first=True)`：把一组**变长**序列
  填充到最大长度，拼成 batch

典型流程：先 `split` 出变长片段，再 `pad_sequence` 统一形状，
后面再接卷积/注意力（配合掩码去掉填充位）。

示例：

```python
import torch
import torch.nn as nn

# 1. split：按长度列表切分（如 2688 = 2048 + 640）
emb = torch.randn(3, 2688)
main, multi = torch.split(emb, [2048, 640], dim=-1)
print('split by sizes:', main.shape, multi.shape)

# 2. chunk：均分成 3 份
ch = torch.randn(1, 6, 32)
print('chunk(3, dim=1):', [c.shape for c in ch.chunk(3, dim=1)])

# 3. pad_sequence：3 个长度分别为 5/3/4 的序列 -> [3, 5, 8]
chunks = [torch.randn(5, 8), torch.randn(3, 8), torch.randn(4, 8)]
padded = nn.utils.rnn.pad_sequence(chunks, batch_first=True)
print('pad_sequence:', padded.shape)
```

程序输出结果:

```python
split by sizes: torch.Size([3, 2048]) torch.Size([3, 640])
chunk(3, dim=1): [torch.Size([1, 2, 32]), torch.Size([1, 2, 32]), torch.Size([1, 2, 32])]
pad_sequence: torch.Size([3, 5, 8])
```

## 构建注意力边界：cu_seqlens

FlashAttention 等高效注意力后端要求输入 `cu_seqlens`——每个序列的
**token 区间边界**（前缀和数组，头部补 0）。构建它的标准套路是：

`torch.repeat_interleave(每序列token数, 重复次数) → cumsum → F.pad(头部补0)`

以「2 个视频：视频 0 有 2 帧（每帧 3×4=12 token），视频 1 有 1 帧（2×2=4 token）」为例：

```python
import torch
import torch.nn.functional as F

grid_thw = torch.tensor([[2, 3, 4], [1, 2, 2]], dtype=torch.int32)  # [帧数, 高, 宽]
rep = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
print('repeat_interleave:', rep)     # 每帧 token 数：12, 12, 4
print('cumsum:', rep.cumsum(dim=0, dtype=torch.int32))   # 前缀和
print('F.pad:', F.pad(rep.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0))  # 边界
```

程序输出结果:

```python
repeat_interleave: tensor([12, 12,  4], dtype=torch.int32)
cumsum: tensor([12, 24, 28], dtype=torch.int32)
F.pad: tensor([ 0, 12, 24, 28], dtype=torch.int32)
```

即三个 token 区间分别是 `[0,12)`、`[12,24)`、`[24,28)`。
若 `repeat_interleave` 在图编译下不可用，可用 `torch.searchsorted` 在
重复计数的前缀和上二分查找出等价索引。

## 掩码抑制与位置定位

- `masked_fill(mask, value)`：把 mask 为 `True` 的位置替换为指定值
  （常用 `-1e9` 填充 logits，使 softmax 后概率≈0，从而**禁止采样某 token**）
- `torch.nonzero(cond)`：返回条件为真的**下标**，用于定位特殊 token 位置
- `any()` / `max()` / `tolist()`：判断存在、取最大值、转 Python 列表

示例：

```python
import torch

# 1. 抑制：把第 3 列的 logits 压到 -1e9
logits = torch.tensor([[1.0, 2.0, -1.0, 0.5], [0.5, -0.5, 3.0, 2.5]])
suppressed = torch.zeros(4, dtype=torch.bool)
suppressed[3] = True
print('masked_fill_:', logits.masked_fill(suppressed.unsqueeze(0), -1e9))

# 2. 定位：找出等于 151667 的位置
ids = torch.tensor([0, 151667, 5, 151667, 151667])
print('nonzero:', torch.nonzero(ids == 151667).squeeze())

# 3. 判断与取值
print('any:', torch.tensor([0, 0, 1]).any(),
      '| max:', torch.tensor([1, 5, 3]).max(),
      '| tolist:', torch.arange(3).tolist())
```

程序输出结果:

```python
masked_fill_: tensor([[ 1.0000e+00,  2.0000e+00, -1.0000e+00, -1.0000e+09],
        [ 5.0000e-01, -5.0000e-01,  3.0000e+00, -1.0000e+09]])
nonzero: tensor([1, 3, 4])
any: tensor(True) | max: tensor(5) | tolist: [0, 1, 2]
```

## 组合查表：Embedding 偏移合并

当数据由**多组码本**组成（如 8 层 RVQ 语音码），可以把它们合并进**一张**
Embedding 表，用「偏移量」区分组别：

1. `offset = arange(组数).view(1, -1, 1) * 码本大小`：各组的词表偏移
2. `codes + offset`：把各组码映射到全局词表下标
3. 查表后对组维度 `mean(1)`：得到每个位置的联合向量

示例（码本大小 1024、8 组、向量维 2048）：

```python
import torch
import torch.nn as nn

emb = nn.Embedding(1024 * 8, 2048)         # 合并后的全局码表
codes = torch.tensor([[0, 1], [2, 3]])      # [batch, seq] 两层码
offset = torch.arange(8).view(1, -1, 1) * 1024   # [1, 8, 1]
hidden = emb(codes.unsqueeze(1) + offset).mean(1)  # 查表 + 层平均
print('合并查表+层平均:', hidden.shape)      # [batch, seq, 2048]
```

程序输出结果:

```python
合并查表+层平均: torch.Size([2, 2, 2048])
```

## 向上取整与分块：torch.ceil

`torch.ceil(x).long()` 向上取整并转整数，常用于**分块计算**：
给定总长度与每块大小，求需要多少块（余数也按一整块算）。

示例：

```python
import torch

feature_lens = torch.tensor([100, 500, 1000])
n_window = 50
print('ceil:', torch.ceil(feature_lens / (n_window * 2)).long())   # 每块 100
```

程序输出结果:

```python
ceil: tensor([ 1,  5, 10])
```

即 100/500/1000 个特征分别需要 1/5/10 个长度为 100 的块（500 恰好整除，
1000 整除为 10）。

# 14. torch 与 NPU：设备与内存管理

> 前面章节的示例都运行在 CPU 上。在昇腾（Ascend）NPU 上使用 PyTorch，需要安装 `torch_npu`：它以扩展方式把 `npu` 设备类型注册进 PyTorch，并提供 `torch.npu.*` 系列接口与一批 CANN 融合算子。本章结合 vllm-omni 的 NPU 平台源码（`vllm_omni/platforms/npu`），讲解 NPU 环境下的设备、内存与数据搬运。

## 认识 NPU 设备

`torch.device` 是 PyTorch 对"设备"的抽象。在安装了 torch_npu 的环境里，可以像下面这样创建 NPU 设备对象：

```python
device = torch.device("npu")              # 默认 0 号 NPU
device0 = torch.device("npu", 0)          # 指定 0 号
device = torch.device("npu", local_rank)  # 按本地序号
```

注意：`npu` 并不是 PyTorch 内置的设备类型，它由 torch_npu 在导入时注册。在未安装 torch_npu 的环境里直接构造会报错（本地 CPU 版 torch 实测）：

```python
import torch

torch.device("npu")
```

程序输出结果（未安装 torch_npu 时）：

```python
RuntimeError: Expected one of cpu, cuda, ipu, ... device type at start of device string: npu
```

安装 torch_npu 后即可正常使用。源码中 platform.py 用 `torch.device("npu", local_rank)` 返回当前设备、用 `torch.npu.device_count()` 查询 NPU 数量；在判断"某个设备/张量是不是 NPU"时，惯用写法是读 `device.type`（`_310p/__init__.py` 的 `is_310p` 就是这种模式）：

```python
def is_npu(device):
    """源码中判断设备是否为 NPU 的惯用写法。"""
    return device is not None and getattr(device, "type", "") == "npu"

t = torch.tensor([1.0])
print("t.device ->", t.device, "| t.device.type ->", t.device.type)
print("is_npu(t.device) ->", is_npu(t.device))
```

程序输出结果：

```python
t.device -> cpu | t.device.type -> cpu
is_npu(t.device) -> False
```

这个判断模式在源码中大量用于"只有 NPU 才做特殊处理"的分支，例如 `patch_step_audio2_hift_for_npu` 中 `self.device.type != "npu"` 为真就跳过补丁、原样执行。

## 内存与设备信息查询

`torch.npu.mem_get_info(device)` 返回 `(free, total)` 两个值（可用显存、总显存，单位字节），`torch.npu.get_device_properties(device_id)` 返回设备属性对象（其中 `total_memory` 为总显存）。platform.py 正是用它们实现显存查询接口：

```python
free, _ = torch.npu.mem_get_info(device)         # 可用显存
free, total = torch.npu.mem_get_info(device)     # 可用 + 总显存
device_props = torch.npu.get_device_properties(device_id)
total_memory = device_props.total_memory         # 总显存字节
```

在本地（无 NPU）环境中，等价的概念可以用张量的字节属性演示。任何张量都有三个与内存相关的属性：`numel()`（元素个数）、`element_size()`（每个元素字节数）、`nbytes`（总字节数 = numel × element_size）：

```python
import torch

x = torch.randn(2, 3, dtype=torch.bfloat16)
print("x:", x.shape, x.dtype)
print("numel:", x.numel(), "| element_size:", x.element_size(), "| nbytes:", x.nbytes)
print("总字节(公式):", x.numel() * x.element_size())
```

程序输出结果：

```python
x: torch.Size([2, 3]) torch.bfloat16
numel: 6 | element_size: 2 | nbytes: 12
总字节(公式): 12
```

## 设备同步与流

NPU 与 GPU 一样采用异步执行模型：算子入队后立即返回，真正执行发生在设备端。需要等待时调用 `torch.npu.synchronize()`；需要与流配合时，用 `torch.npu.current_stream()` 取当前流，用 `torch.npu.stream(stream)` 进入某个流的上下文。

源码 `record_device_event` 的注释揭示了 NPU 流的一个关键差异：HCCL 分布式通信可能使用默认流不可见的内部流，因此要先同步默认流、再记录事件，才能保证 `d2h_stream.wait_event(event)` 捕获到完整输出。

```python
torch.npu.current_stream().synchronize()   # 先同步默认流
event = torch.Event()                      # 再记录事件（见下节）
event.record()
```

在 npu_ar_model_runner.py 中，异步状态更新使用流上下文，避免阻塞主线程：

```python
with torch.npu.stream(global_stream()):
    global_stream().wait_event(self.sampling_done_event)   # 等采样事件完成
    self._update_states_after_model_execute(...)
```

## 事件：torch.Event 与 npu.Event 的兼容性

事件（Event）用于跨流同步。在 NPU 上事件有两类写法，源码注释里记录了一个很实用的经验：

- `torch.npu.Event()`：NPU 原生事件，但 torch_npu 2.10 下不能被公开的 `torch.Stream` 包装流消费；
- `torch.Event()`：标准事件，会被分派到当前激活的 NPU 后端，与包装流双向兼容。

因此 platform.py 的 `record_device_event` 返回 `torch.Event()`；而 npu_ar_model_runner.py 在只需要"采样完成"标记时使用 `torch.npu.Event()`：

```python
if self.sampling_done_event is None:
    self.sampling_done_event = torch.npu.Event()
self.sampling_done_event.record()   # 采样完成后记录
```

判断标准很简单：**需要与流/异步输出配合时用 torch.Event()，只做设备内同步标记时用 torch.npu.Event()。**

## 数据类型与统一迁移

NPU 上模型通常以半精度（float16/bfloat16）运行。把模块或张量迁移到目标设备与类型用统一的 `.to()`：`tensor.to(device, dtype)`、`module.to(device=device, dtype=dtype)`。qwen3_tts.py 加载完权重后就是这样统一迁移解码器的：

```python
self.decoder.to(device=device, dtype=runtime_dtype)
```

实测 `.to()` 的类型迁移与就地拷贝（本地 CPU 演示）：

```python
import torch

y = torch.randn(2, 2)
y2 = y.to(dtype=torch.float16)
print("y.to(float16):", y2.dtype, y2)

dst = torch.empty_like(y)
dst.copy_(y, non_blocking=True)        # 就地拷贝，non_blocking 允许异步
print("copy_ 后相等:", torch.equal(dst, y))
```

程序输出结果：

```python
y.to(float16): torch.float16 tensor([[ 0.0188,  0.0442],
        [ 0.3818, -0.0970]], dtype=torch.float16)
copy_ 后相等: True
```

混合精度方面，NPU 平台覆写了 `create_autocast_context`：`device_type == "npu"` 时优先尝试 `torch.npu.amp.autocast(dtype=dtype)`，一旦该 API 在当前 torch_npu 版本不可用，就回退为空上下文（nullcontext），保证流程不被中断：

```python
try:
    return torch.npu.amp.autocast(dtype=dtype)
except (RuntimeError, TypeError, ValueError) as exc:
    logger.warning("autocast unavailable: %s", exc)
return nullcontext()
```

## 零拷贝内存池：data_ptr 与 frombuffer

在多机多卡推理（如 vllm-omni 的分阶段部署）中，stage 之间要搬运大量张量。yuanrong_transfer_engine_connector.py 的做法是：在 NPU 上预先分配一块大内存池，用 `data_ptr()` 取出裸地址注册给传输引擎，之后 RDMA 直接读写这块内存，收发双方都不再拷贝。

```python
pool = torch.empty(pool_size, dtype=torch.uint8, device="npu")   # NPU 字节池
base_ptr = int(pool.data_ptr())                                  # 裸地址，注册给传输引擎
```

发送时，把任意张量以字节视图拷入池中：`data.view(torch.uint8).flatten()` 把浮点张量重新解释为字节序列，`copy_(..., non_blocking=...)` 就地写入；接收时用 `torch.frombuffer` 从字节缓冲直接构造张量视图。这些技巧与设备无关，本地即可演示：

```python
import torch

# 1) 内存池：一整块连续字节
pool = torch.empty(16, dtype=torch.uint8)
print("pool.data_ptr() ->", pool.data_ptr())

# 2) 字节视图：把张量当字节流看，不复制数据
view = pool.view(torch.uint8).flatten()
print("view(uint8).flatten():", view,
      "| 同存储:", view.data_ptr() == pool.data_ptr())

# 3) 从 bytes 构造只读张量（接收方向）
raw = bytes(range(16))
t = torch.frombuffer(raw, dtype=torch.uint8)
print("frombuffer:", t)
```

程序输出结果：

```python
pool.data_ptr() -> 2446017495232
view(uint8).flatten(): tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8) | 同存储: True
frombuffer: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15], dtype=torch.uint8)
```

三个要点：`data_ptr()` 只是裸地址，内存生命周期仍由张量对象管理；`view(torch.uint8)` 不产生新存储，跨设备拷贝前需先 `contiguous()`；`copy_` 的 `non_blocking` 只在源与目标位于不同设备时才有异步意义。

# 15. torch 与 NPU：融合算子与布局适配

> 昇腾 NPU 的算子由 CANN 提供。`torch_npu` 把一批"融合算子"（fused op）挂载到 `torch_npu` 命名空间，例如 `npu_rotary_mul`、`npu_rms_norm`、`npu_swiglu`、`npu_fused_infer_attention_score_v2` 等。它们把多个 PyTorch 算子合并进一个 CANN 内核，减少中间张量与内核启动次数。本章从 vllm-omni 的 NPU 补丁代码出发，讲解最常见的替换与适配模式。

## 融合算子：为什么替换

同一功能，PyTorch 写法与 NPU 融合算子的对应关系：

| 功能 | PyTorch 写法 | NPU 融合算子 |
|---|---|---|
| 旋转位置编码 | 手工切片、cos/sin 乘加 | torch_npu.npu_rotary_mul |
| RMS 归一化 | x * rsqrt(mean(x²)+eps) * w | torch_npu.npu_rms_norm |
| 残差 + RMSNorm | 先相加再归一化 | torch_npu.npu_add_rms_norm |
| SwiGLU 激活 | silu(gate) * up | torch_npu.npu_swiglu |
| 融合注意力 | F.scaled_dot_product_attention | npu_fused_infer_attention_score_v2 |

替换动机主要有三个：减少内核启动（几个算子合成一个）；省掉中间张量的读写；部分功能只有融合算子支持（如 310P 上的 flash attention）。接下来逐个看源码中的真实用法。

## 融合 RoPE：npu_rotary_mul 与布局回退

旋转位置编码（RoPE）是 Transformer 的标配。源码 layers/rotary_embedding.py 封装了一个带"布局回退"的融合 RoPE：昇腾的 `npu_rotary_mul` 对 BNSD 布局（batch, heads, seq, dim）有形状限制（要求 `head_dim//2` 能被 `32/element_size` 整除、且 `batch*num_heads <= seq_len*8`），不满足时就把张量转成 BSND 布局再算，算完转回：

```python
def npu_rotary_mul_with_bsnd_fallback(hidden_states, cos, sin, unsqueeze_dim):
    if unsqueeze_dim != 1 or _bnsd_rotary_shape_is_supported(hidden_states):
        return torch_npu.npu_rotary_mul(
            hidden_states, cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim))
        
    # 回退：BNSD -> BSND 计算 -> 转回
    hidden_states_bsnd = hidden_states.transpose(1, 2).contiguous()
    output_bsnd = torch_npu.npu_rotary_mul(
        hidden_states_bsnd, cos.unsqueeze(2), sin.unsqueeze(2))
    return output_bsnd.transpose(1, 2)
```

其中 `unsqueeze_dim=1` 表示输入是 BNSD 布局（Transformers 的 RoPE 约定：在 heads 维插入一个维度以便与 cos/sin 广播）。核心是 `transpose(1, 2).contiguous()`：`transpose` 只交换视图（内存不变），`contiguous()` 才真正重排内存。本地演示布局转换与数据保真：

```python
import torch

x = torch.arange(2 * 4 * 3 * 8).reshape(2, 4, 3, 8).float()   # B,N,S,D
print("BNSD 输入:", x.shape)
bsnd = x.transpose(1, 2).contiguous()                        # -> B,S,N,D
print("transpose(1,2).contiguous() -> BSND:", bsnd.shape)
back = bsnd.transpose(1, 2)                                  # 转回 BNSD
print("转回 BNSD:", back.shape, "| 数据一致:", torch.equal(back, x))
```

程序输出结果：

```python
BNSD 输入: torch.Size([2, 4, 3, 8])
transpose(1,2).contiguous() -> BSND: torch.Size([2, 3, 4, 8])
转回 BNSD: torch.Size([2, 4, 3, 8]) | 数据一致: True
```

另一个 RoPE 变体是 310P 补丁里的 `torch_npu.npu_apply_rotary_pos_emb(q, k, cos, sin, rotary_mode="half")`——它一次处理 Q、K 两个张量，且要求输入是连续的 BSND 布局，所以调用前要逐个 `transpose(...).contiguous()`。这类"先转布局、再调算子、再转回"的模式在 NPU 适配中非常常见。

## 融合归一化：npu_rms_norm 与 npu_add_rms_norm

RMSNorm 是很多新模型（如 Qwen）使用的归一化层。qwen3_tts_tokenizer_v2.py 的 NPU 补丁把它的 forward 直接替换为融合算子：

```python
def _rms_norm_forward_npu(self, hidden_states):
    return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
```

注意返回值取 `[0]`：`npu_rms_norm` 返回 `(输出, 均方根)` 二元组，而 PyTorch 的 forward 只需要输出。

310P 补丁里还有一个更进一步的融合——把"残差相加 + RMSNorm"合并为 `torch_npu.npu_add_rms_norm`，正好接在注意力输出之后：

```python
hidden_states, _, residual = torch_npu.npu_add_rms_norm(
    hidden_states,                       # 注意力输出
    residual,                            # 残差
    self.post_attention_layernorm.weight,
    self.post_attention_layernorm.variance_epsilon,
)
```

它返回 `(归一化输出, 新残差, 旧残差)` 三个值，比"先 `hidden_states + residual` 再单独做 RMSNorm"少了一次张量分配和一次内核启动。

## 融合激活：npu_swiglu

SwiGLU 激活是 `silu(gate) * up` 的组合。minimax_h3.py 的 NPU 补丁把"门控投影 + 激活"打包：先用一次线性变换得到拼接好的 `gate_up`，再用 `npu_swiglu` 一步完成激活与相乘：

```python
def npu_swiglu_from_packed(gate_up):
    return torch_npu.npu_swiglu(gate_up, dim=-1)

def _forward_mlp_npu(self, x):
    gate_up = self.gate_up_proj(x)              # 一次 GEMM 得到 [..., 2*hidden]
    return self.down_proj(npu_swiglu_from_packed(gate_up))
```

与之对应的纯 PyTorch 实现（理解原理用）是：

```python
gate, up = gate_up.chunk(2, dim=-1)
return F.silu(gate) * up
```

融合算子把 `chunk + silu + 乘法` 合成一个内核，并且直接吃拼接后的输入，省掉了中间的拆分。

## 注意力适配：掩码形状与 fused attention

NPU 上最典型的适配问题是注意力掩码的形状。CosyVoice2 的 DiT 注意力构造了 `(B, 1, 1, S)` 形状的 key-padding 掩码并直接喂给 `F.scaled_dot_product_attention`，但在昇腾上 SDPA 会被路由到 `npu_fusion_attention`（aclnnFlashAttentionScore），该内核只接受 `[B,N,S,S] / [B,1,S,S] / [1,1,S,S] / [S,S]` 四种形状，`[B,1,1,S]` 会直接报错 161001。cosyvoice2_dit_attn.py 的补丁先把掩码扩展成 `[B,1,S,S]`：

```python
# (B, 1, S) key-padding -> (B, S_q, S_kv)
if attn_mask.dim() == 3 and attn_mask.shape[-2] == 1:
    attn_mask = attn_mask.expand(-1, q_len, -1)
if attn_mask.dim() == 3:
    attn_mask = attn_mask.unsqueeze(1)               # -> (B, 1, S_q, S_kv)
if attn_mask.dim() == 4 and attn_mask.shape[-2] == 1 and kv_len > 1:
    attn_mask = attn_mask.expand(-1, -1, q_len, -1)  # (B,1,1,S) -> (B,1,S,S)
```

本地演示掩码扩展与 SDPA：

```python
import torch
import torch.nn.functional as F

mask = torch.zeros(2, 1, 5)                     # (B,1,S) key-padding
print("原 mask:", mask.shape)
m2 = mask.expand(-1, 5, -1).unsqueeze(1)        # -> (B,1,Sq,S)
print("expand+unsqueeze ->", m2.shape)

q = torch.randn(2, 4, 5, 8)
k = torch.randn(2, 4, 5, 8)
out = F.scaled_dot_product_attention(q, k, k, attn_mask=m2)
print("SDPA 输出:", out.shape)
```

程序输出结果：

```python
原 mask: torch.Size([2, 1, 5])
expand+unsqueeze -> torch.Size([2, 1, 5, 5])
SDPA 输出: torch.Size([2, 4, 5, 8])
```

除了改掩码形状，还可以直接绕过融合内核：`torch.nn.attention.sdpa_kernel(SDPBackend.MATH)` 把 SDPA 强制切到 MATH 后端（纯 PyTorch 数学实现），用于"融合内核不可用或出错"时的兜底：

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel(SDPBackend.MATH):
    out = F.scaled_dot_product_attention(q, k, k)
    print("MATH 后端 SDPA:", out.shape)
```

程序输出结果：

```python
MATH 后端 SDPA: torch.Size([2, 4, 5, 8])
```

另外两个注意力适配点：
- **GQA**：minimax_h3.py 在调用 SDPA 前校验 `num_heads % num_kv_heads == 0`，并通过 `enable_gqa=num_heads != num_key_value_heads` 打开 NPU 原生 GQA 支持；
- **融合注意力算子**：kv_quant_npu.py 使用 `torch_npu.npu_fused_infer_attention_score_v2(q, k, v, input_layout=..., num_query_heads=..., softmax_scale=..., ...)`，直接传入量化后的 Q/K/V 与反量化系数，一次调用完成"注意力 + 反量化"（见下节）。

## FP8 量化注意力：float8_e4m3fn 与 QuaRot

NPU 的 FP8 注意力链路（kv_quant_npu.py）展示了算子组合的完整流程：**Hadamard 旋转 → 分块量化 → 融合注意力（带反量化）**。Q/K 先与预计算的旋转矩阵做矩阵乘（QuaRot 风格预处理，降低离群值影响）：

```python
rot = create_rot(QuaRotMode.HADAMARD, head_dim, seed=425500).to(device=device, dtype=dtype)
q_f = torch.matmul(query, rot)
k_f = torch.matmul(key, rot)
```

再用 `fa_block_quant_preprocess` 把每个 128/256 大小的块量化到 FP8（`torch_npu.float8_e4m3fn`），得到量化张量与缩放系数：

```python
q, q_scale = fa_block_quant_preprocess(q_f, block_size=128,
                                       dst_type=torch_npu.float8_e4m3fn, layout=layout)
v, v_scale = fa_block_quant_preprocess(value, block_size=256,
                                       dst_type=torch_npu.float8_e4m3fn, layout=layout)
```

最后把量化张量和反量化系数一起交给融合注意力算子，`query_quant_mode=7` 表示走"分块 FP8 反量化"路径，输出还原为原始 dtype：

```python
out = torch_npu.npu_fused_infer_attention_score_v2(
    q, k, v, input_layout=layout,
    num_query_heads=n, softmax_scale=scale,
    pre_tokens=2147483647, next_tokens=2147483647,   # 不截断上下文
    query_quant_mode=7, key_quant_mode=7, value_quant_mode=7,
    dequant_scale_query=q_scale, dequant_scale_key=k_scale,
    dequant_scale_value=v_scale, out_dtype=out_dtype,
)[0]
```

`pre_tokens/next_tokens=2147483647`（INT32_MAX）表示不限制注意力窗口；返回后还会按需裁剪输出序列长度（`out[:, :, :s, :]`），因为融合内核可能对序列长度做了对齐填充。

## 权重布局：ND / FRACTAL_NZ / FRACTAL_Z

昇腾矩阵乘对权重有特殊的存储布局要求：默认的 ND（按行连续）布局之外，还有分形布局 `FRACTAL_NZ`（矩阵乘用）与 `FRACTAL_Z`（卷积用）。权重量化后通常要转成 NZ 布局才能喂给融合内核。vllm-ascend 提供 `maybe_trans_nz` / `nd_to_nz_2d`，torch_npu 提供 `npu_format_cast(tensor, format)` 做显式格式转换：

```python
# Linear 权重：转成 FRACTAL_NZ（矩阵乘布局）
module.weight.data = maybe_trans_nz(module.weight.data)
# Conv 权重：显式转成 FRACTAL_Z（卷积布局），先保证连续
module.weight.data = torch_npu.npu_format_cast(
    module.weight.data.contiguous(), _ACL_FORMAT_FRACTAL_Z)
```

平台层还有两个配套开关：
- `torch.npu.config.allow_internal_format = True`：允许算子内部保留 NZ 存储格式，避免量化权重反复转回 ND（platform.py 对 LLM 开启）；
- `torch.npu.config.allow_internal_format = False` + `torch.npu.set_compile_mode(jit_compile=False)`：Code2Wav 这类普通卷积链路要求 ND 布局且禁用 JIT，保证可被图捕获（见第 16 章）。

`contiguous()` 在格式转换前几乎是必须的：转置、切片产生的张量内存不连续，直接转格式会失败或产生意外拷贝。本地演示"视图不连续 → contiguous 重排"：

```python
import torch

x = torch.arange(12.0).reshape(3, 4)
view = x.t()                          # 转置视图：内存不连续
print("转置视图 is_contiguous:", view.is_contiguous())
y = view.contiguous()                 # 重排内存
print("contiguous 后 is_contiguous:", y.is_contiguous())
```

程序输出结果：

```python
转置视图 is_contiguous: False
contiguous 后 is_contiguous: True
```

此外 310P 的 flash attention 要求 token 数量按 16 对齐，源码用 `aligned_16` 补齐并用 `seq_lens` 携带真实长度（见第 16 章降级策略）。

# 16. torch 与 NPU：图模式、分布式与降级策略

> NPU 推理性能的关键是**图执行**（ACL Graph）：把一串算子编译成一张图，一次下发、按固定形状重放，省掉每步的调度开销。本章结合 graph_tools.py 与 310P 补丁，讲解 NPU 上的图捕获/重放、HCCL 分布式，以及算子不可用时的各种降级手段。

## NPU 图捕获：NPUGraph 与 graph_tools

`torch.npu` 提供了与 CUDA Graph 对应的图 API：`torch.npu.NPUGraph()` 创建图对象，`npu.graph(graph, pool=...)` 上下文内执行的算子会被捕获进图，之后 `graph.replay()` 按固定输入形状重放。graph_tools.py 的 NPUExactGraphRunner 是完整的生产级封装，核心流程是：

```python
npu = torch.npu
npu.synchronize()                        # 捕获前同步
graph = npu.NPUGraph()
with torch.inference_mode(), npu.graph(graph, pool=self._graph_pool):
    static_outputs = compute(*static_inputs)   # 用静态输入跑一遍
npu.synchronize()
```

要点：
- 捕获前必须 `npu.synchronize()`，确保设备空闲、状态干净；
- 捕获必须在 `torch.inference_mode()` 下进行，图内不追踪梯度；
- 图与**精确的输入签名**绑定：`(shape, dtype, device)` 三元组作为缓存 key，形状一变就得另建一张图。源码里 `_tensor_signature` 返回 `(tuple(shape), str(dtype), str(device))`；
- 用 `npu.is_current_stream_capturing()` 判断当前流是否正在捕获，避免"图里套图"；
- 捕获失败后整个进程的分配器/RNG 状态可能已损坏，必须重启（源码会直接抛错提示）。

## 图输入输出的静态化：copy_ / detach / clone

图一旦捕获，输入输出都是**固定内存**。每次重放前要把本次的真实数据 `copy_` 进静态输入缓冲；图输出是持久内存、会被下一次重放覆盖，所以在返回前要 `detach().clone()` 一份：

```python
def replay(self, inputs):
    with torch.inference_mode():
        for static, current in zip(self.static_inputs, inputs):
            static.copy_(current)              # 当前数据拷入静态缓冲
        self.graph.replay()                    # 重放
        # 输出是持久内存，克隆一份再返回，避免被下次重放覆盖
        return tuple(output.detach().clone() for output in self.static_outputs)
```

这套"静态缓冲 + copy_ + detach().clone()"与设备无关，本地可以完整演示：

```python
import torch

static_in = torch.zeros(3)               # 静态输入缓冲
static_out = torch.zeros(3)              # 模拟图输出（持久内存）
current = torch.tensor([1.0, 2.0, 3.0])

with torch.inference_mode():
    static_in.copy_(current)             # 真实数据入缓冲
    static_out = static_in * 2           # 模拟图内计算
    result = static_out.detach().clone() # 取走一份，防止下次重放覆盖
print("图输出（持久内存）:", static_out)
print("返回给调用方:", result, "| 不追踪梯度:", result.requires_grad)
```

程序输出结果：

```python
图输出（持久内存）: tensor([2., 4., 6.])
返回给调用方: tensor([2., 4., 6.]) | 不追踪梯度: False
```

310P 补丁里还有一个与之配合的实践：`_ensure_buffers` 预分配一批固定 buffer（`proj_buf`、`all_codes`），循环内用 `zero_()` 清零、`copy_()` 写入、`torch.cat` 拼装输入，确保每次重放的数据都在预分配内存里，不触发新的分配。

## 分布式：HCCL 与集合通信

NPU 上的分布式通信后端是 **HCCL**（华为集合通信库），在平台类里声明 `dist_backend: str = "hccl"`。fused_moe.py 展示了两个常用的 torch 分布式接口：

```python
backend = torch.distributed.get_backend(world_group.device_group)    # 查当前通信后端
torch.distributed.all_gather_object(gathered_list, value, group=...) # 收集 Python 对象
```

`all_gather_object` 用于收集各 rank 的 Python 值（例如各 rank 的 MoE token 数），`all_gather` 才是张量版。NPU 上还有 MC2（多卡通信）专用通信组，vllm-ascend 通过 `vllm_ascend.distributed.parallel_state._MC2` 保存；`prepare_fused_moe_runtime` 在缺失时才初始化它：

```python
vllm_ascend_parallel_state._MC2 = vllm_init_model_parallel_group(
    group_ranks, local_rank, backend, group_name="mc2")
```

在跨设备数据搬运场景（yuanrong_transfer_engine_connector.py），`_sync_tensor_device` 演示了一个通用技巧：通过 `getattr(torch, device.type)` 动态拿到设备后端模块（npu 就是 torch.npu，cuda 就是 torch.cuda），再调用它的 `current_stream(device).synchronize()` 或 `synchronize()`：

```python
backend = getattr(torch, device.type, None)     # torch.npu / torch.cuda / ...
current_stream = getattr(backend, "current_stream", None)
current_stream(device).synchronize()            # 同步该设备当前流
```

本地演示 `getattr(torch, device.type)` 的分派逻辑：

```python
backend = getattr(torch, "cpu", None)
print("getattr(torch, 'cpu') ->", backend)
```

程序输出结果：

```python
getattr(torch, 'cpu') -> <module 'torch.cpu' from '...torch\\cpu\\__init__.py'>
```

## 算子不可用时的降级策略

昇腾某些算子缺失或行为不符时，源码里主要有四类降级手段。这也是学习 NPU 适配最重要的实战经验。

**1. 局部算子回退 CPU。** A5 设备不支持 `torch.stft`，speaker 编码器的 mel 前端整体移到 CPU 计算再搬回 NPU；Step-Audio2 的 `_f02sine` 在脉冲模式或降采样不被支持时也回退 CPU：

```python
output_device = f0_values.device
output = self._step_audio2_original_f02sine(f0_values.cpu())   # CPU 算
return output.to(output_device)                                # 搬回 NPU
```

**2. 用纯 PyTorch 算子替换有问题的内核。** Step-Audio2 的 HiFT 用 480 倍 `F.interpolate(mode="linear")` 降采样，在昇腾 `linear1d` 内核上会崩溃（ACL 507015）。补丁利用"偶数整数倍降采样时每个输出点恰好在两个源样本中点"的性质，用切片与均值替代：

```python
def _linear_downsample_even_scale(x, scale):
    left, right = scale // 2 - 1, scale // 2
    return (x[..., left::scale] + x[..., right::scale]) * 0.5
```

本地验证该中点公式（scale=4，输入 12 点 → 输出 3 点）：

```python
import torch

x = torch.arange(12.0).reshape(1, 1, 12)
scale = 4
left, right = scale // 2 - 1, scale // 2
mid = (x[..., left::scale] + x[..., right::scale]) * 0.5
print("原:", x)
print("中点降采样:", mid)
```

程序输出结果：

```python
原: tensor([[[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.]]])
中点降采样: tensor([[[1.5000, 5.5000, 9.5000]]])
```

**3. 强制/禁用编译路径。** 三条典型手段：`torch.npu.set_compile_mode(jit_compile=False)` 关闭 JIT（图捕获要求）；`torch._dynamo.disable(orig)` 摘掉 `@torch.compile` 装饰器（CosyVoice2 的 Dynamo 追踪在昇腾 fake-tensor 上崩溃，直接跑 eager）；`sdpa_kernel(SDPBackend.MATH)` 强制 SDPA 走数学后端（见第 15 章）。`torch._dynamo.disable` 本地可演示：

```python
import torch

def fn(x):
    return x * 2

orig = torch._dynamo.disable(fn)     # 返回未编译的原始函数
print("torch._dynamo.disable 返回:", orig(3))
```

程序输出结果：

```python
torch._dynamo.disable 返回: 6
```

**4. 在 CPU 上做采样前处理。** 310P 的采样算子和 grammar bitmask 无法进图，所以采样逻辑在 CPU 完成再搬回设备。npu_ar_model_runner.py 的做法：先把 logits 转成 CPU float32，应用 bitmask，再转回设备与原 dtype：

```python
logits_dtype = logits.dtype
logits = logits.to("cpu").float()        # CPU + float32
apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)
logits = logits.to(self.device).to(logits_dtype)   # 搬回设备
```

配合的还有 token 越界保护 `clamp(max=logits_vocab)`——采样头词表比文本 tokenizer 窄时，把越界 id 截断：

```python
ids = torch.tensor([0, 5, 200, 150000])
print("clamp(max=128):", ids.clamp(max=128))
```

程序输出结果：

```python
clamp(max=128): tensor([  0,   5, 128, 128])
```

降级策略的通用原则：**优先保持主路径在 NPU 上（融合算子、图模式），只有真正不支持的算子才局部回退，且回退范围越小越好**——例如 Step-Audio2 只把 `_f02sine` 挪到 CPU，而不是整个 HiFT。

## 310P 采样循环：张量组合实战

`_310p/patch/qwen3_tts.py` 的 CodePredictor forward 循环是 NPU 场景下张量组合的集中展示：`torch.full` 构造批次一致的常量张量、`torch.arange + unsqueeze + expand` 铺满整批位置、`F.softmax(dtype=torch.float32)` 稳定采样、`logits.argmax` 贪心解码、`F.embedding` 查表取嵌入。全部可本地运行：

```python
import torch
import torch.nn.functional as F

bsz, num_groups, seq = 2, 4, 5

# 1) 批次一致的标量 -> 张量（top_k / seq_lens）
top_k = torch.full((bsz,), 50, dtype=torch.int32)
seq_lens = torch.full((bsz,), seq, dtype=torch.int32)
print("top_k:", top_k, "| seq_lens:", seq_lens)

# 2) 位置 id：arange + unsqueeze + expand 铺满整批
pos_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
print("pos_ids:", pos_ids.tolist())

# 3) 每步采样：softmax 归一化 + argmax 贪心
logits = torch.randn(bsz, num_groups) * 0.5
probs = F.softmax(logits, dim=-1, dtype=torch.float32)
print("softmax 每行和:", probs.sum(dim=-1))
code = logits.argmax(dim=-1, keepdim=True)
print("argmax:", code.tolist())

# 4) 查表取嵌入
emb = torch.randn(4, 6)
idx = torch.tensor([0, 2])
print("F.embedding:", F.embedding(idx, emb).shape)
```

程序输出结果：

```python
top_k: tensor([50, 50], dtype=torch.int32) | seq_lens: tensor([5, 5], dtype=torch.int32)
pos_ids: [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
softmax 每行和: tensor([1., 1.])
argmax: [[1], [3]]
F.embedding: torch.Size([2, 6])
```

在真实代码里，这些组合被进一步约束在 NPU 图模式下：`proj_buf[:padded_bsz].zero_()` 先清零、`torch.cat` 拼装初始嵌入、`proj_buf.copy_(projection(...))` 写入预分配 buffer，然后才进入图重放循环——每一步都对应本笔记第 12、13 章讲过的张量操作。