import os  # 导入os模块，用于与操作系统交互，例如读取环境变量、路径操作等
from random import random

import torch  # 导入PyTorch库，一个用于深度学习的开源库

import torch.nn.functional as F  # 导入PyTorch中的神经网络功能函数库

import sys  # 导入sys模块，用于与Python解释器交互，例如修改模块搜索路径

sys.path.append('./models')  # 将'./models'目录添加到Python的模块搜索路径中，这样可以直接导入该目录下的模块
#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import numpy as np  # 导入NumPy库，用于数值计算

from datetime import datetime  # 从datetime模块中导入datetime类，用于处理日期和时间


from wtmamba import net

from torchvision.utils import make_grid  # 从torchvision.utils模块中导入make_grid函数，用于将多张图片组合成网格图片

from data_cod import get_loader, test_dataset  # 从data_cod模块中导入get_loader和test_dataset函数或类

from utils import clip_gradient, adjust_lr  # 从utils模块中导入clip_gradient和adjust_lr函数，分别用于梯度裁剪和学习率调整

from tensorboardX import SummaryWriter  # 导入tensorboardX库中的SummaryWriter类，用于记录训练过程中的数据并可视化

import logging  # 导入logging模块，用于记录日志信息

import torch.backends.cudnn as cudnn  # 导入PyTorch中的cudnn模块，用于设置CUDA相关的配置

from options_cod1 import opt  # 从options_cod模块中导入opt对象，通常用于存储命令行参数或配置文件中的选项
def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    # 计算预测值与掩码的重叠部分（交集）
    inter = (pred * mask).sum(dim=(2, 3))
    # 计算预测值与掩码的总和（并集）
    union = (pred + mask).sum(dim=(2, 3))

    # 计算IoU（Intersection over Union，交并比）
    iou = 1 - (inter + 1) / (union - inter + 1)
    # 返回IoU损失的均值，通过求所有batch中IoU的平均值得到
    return iou.mean()



# 设置cuDNN为benchmark模式，该模式会针对当前配置寻找最适合的卷积算法，以提升运算速度
cudnn.benchmark = True

# 从opt对象中获取RGB图像的根目录
image_root = opt.rgb_root

# depth_root =opt.rgb_d

# 从opt对象中获取真实标签（ground truth）的根目录
gt_root = opt.gt_root

# 从opt对象中获取测试RGB图像的根目录
test_image_root = opt.test_rgb_root

# 从opt对象中获取测试真实标签的根目录
test_gt_root = opt.test_gt_root

# 从opt对象中获取保存路径
save_path = opt.save_path

# 配置日志记录器，将日志信息写入文件
logging.basicConfig(
    filename=save_path + 'DFCNet.log',  # 日志文件路径
    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',  # 日志格式
    level=logging.INFO,  # 最低记录级别为INFO
    filemode='a',  # 文件打开模式为追加
    datefmt='%Y-%m-%d %I:%M:%S %p'  # 日期格式
)

# 记录一条日志信息，表明开始训练PRNet模型
logging.info("DFCNet")
seed=11
seed_everything(seed)
model = net()

# 初始化一个变量来存储模型参数的总数
num_parms = 0

# 检查opt对象中的load属性是否不为None，即是否指定了要加载的模型路径
if (opt.load is not None):
    # 如果指定了模型路径，则调用模型的load_pre方法加载预训练模型
    model.load_pre(opt.load)
    # 打印出加载模型的路径
    # print('load model from ', opt.load)

# 遍历模型的所有参数
for p in model.parameters():
    # 将每个参数的元素数量（即参数的维度乘积）累加到num_parms中
    num_parms += p.numel()

# 使用logging记录模型的总参数数量，以供后续参考
logging.info("Total Parameters (For Reference): {}".format(num_parms))

# 打印出模型的总参数数量，以供在终端查看
# print("Total Parameters (For Reference): {}".format(num_parms))

# 获取模型的所有参数，准备用于优化器
params = model.parameters()

# 使用Adam优化器，并将模型参数和学习率传入
optimizer = torch.optim.Adam(params, opt.lr)

# 设置保存路径
# 检查save_path指定的目录是否存在
if not os.path.exists(save_path):
    # 如果目录不存在，则创建该目录
    os.makedirs(save_path)

# 加载数据
# print('load data...')

# 调用get_loader函数，传入图像根目录、真实标签根目录、批次大小、训练集大小等参数，
# 返回一个用于训练的数据加载器train_loader

# 调用test_dataset函数，传入测试图像根目录、测试真实标签根目录和训练集大小，
# 返回一个用于测试的数据集test_loader
test_loader = test_dataset(test_image_root, test_gt_root, opt.trainsize)
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)

# 计算总的训练步骤数，即训练集大小除以批次大小
total_step = len(train_loader)

# 使用logging记录配置信息
logging.info("Config")

# 记录详细的配置信息，包括训练轮数、学习率、批次大小、训练集大小、梯度裁剪值、学习率衰减率、
# 是否加载预训练模型、保存路径以及学习率衰减的轮数
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# 设置损失函数
# BCEWithLogitsLoss是带logits的二元交叉熵损失函数，通常用于sigmoid层的输出
CE = torch.nn.BCEWithLogitsLoss()

# BCELoss是二元交叉熵损失函数
ECE = torch.nn.BCELoss()

# 初始化步骤计数器
step = 0

# 创建一个用于记录训练过程中各种指标（如损失、准确率等）的SummaryWriter对象
writer = SummaryWriter(save_path + 'summary')

# 初始化最佳平均绝对误差为较大的值（例如1），用于后续寻找更优的模型
best_mae = 1

# 初始化最佳训练轮数为0，用于记录达到最佳平均绝对误差时的训练轮数
best_epoch = 0


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    # 定义train函数，用于训练模型。
    # 参数：
    # train_loader：数据加载器，用于获取训练数据。
    # model：待训练的模型。
    # optimizer：优化器，用于更新模型参数。
    # epoch：训练轮数。
    # save_path：模型保存路径。

    # 使用全局变量step
    global step

    # 将模型移至CUDA设备上，如果可用的话
    model.cuda(1)

    # 将模型设置为训练模式，通常这会启用如dropout和batch normalization等层的训练时行为
    model.train()

    # 初始化sal_loss_all变量，用于累加saliency loss（显著度损失）
    sal_loss_all = 0

    # 初始化loss_all变量，用于累加总的损失
    loss_all = 0

    # 初始化epoch_step变量，用于记录当前epoch的训练步骤数
    epoch_step = 0

    try:
        # 尝试执行训练循环，如果发生异常，则捕获异常处理
        for i, (images, gts) in enumerate(train_loader, start=1):
            # 遍历训练数据加载器train_loader，返回图像和真实标签的批次数据
            # i是批次索引，从1开始计数
            # images是输入图像批次，gts是对应的真实标签批次

            optimizer.zero_grad()
            # 清除优化器中之前的梯度信息

            images = images.cuda(1)
            # 将图像数据移至CUDA设备（GPU）上

            gts = gts.cuda(1)
            # 将真实标签数据移至CUDA设备（GPU）上

            s1= model(images)
            # 将图像数据传入模型，得到四个输出s1, s2, s3, s4
            # 这可能是模型的不同层或不同分支的输出
            bce_iou1 = CE(s1, gts) + iou_loss(s1, gts)

            # 将四个损失相加，得到深度监督的总损失bce_iou_deep_supervision

            loss = bce_iou1
            loss.backward()
            # 反向传播，计算损失关于模型参数的梯度

            clip_gradient(optimizer, opt.clip)
            # 调用clip_gradient函数，对优化器中的梯度进行裁剪，防止梯度爆炸
            # opt.clip是梯度裁剪的阈值

            optimizer.step()
            # 根据梯度更新模型参数

            step += 1
            # 全局变量step递增，记录总的训练步骤数

            epoch_step += 1
            # epoch_step递增，记录当前epoch的训练步骤数

            loss_all += loss.data
            # 累加当前批次的损失到loss_all变量中

            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            # 计算并获取当前GPU上分配的最大内存量（单位：MB）

            # 如果当前迭代次数i是100的倍数，或者是总迭代次数total_step，或者是第一次迭代（i==1）
            if i % 100 == 0 or i == total_step or i == 1:
                # 打印当前时间、当前周期（epoch）、总周期数、当前步骤（step）、总步骤数、当前学习率（LR）和损失（sal_loss）
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||sal_loss:{:4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             optimizer.state_dict()['param_groups'][0]['lr'], loss.data))

                # 在日志中记录训练信息，包括周期、步骤、学习率和损失，以及内存使用情况
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f} , mem_use:{:.0f}MB'.
                    format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'], loss.data,
                           memory_used))

                # 使用tensorboard记录损失值，global_step为当前的步骤数
                writer.add_scalar('Loss', loss.data, global_step=step)

                # 将输入的图像数据转换成网格形式，并归一化，然后添加到tensorboard中
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)

                # 将ground truth数据转换成网格形式，并归一化，然后添加到tensorboard中
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)

        # 计算平均损失值
        loss_all /= epoch_step

        # 在日志中记录平均损失值
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))

        # 使用tensorboard记录每个周期的平均损失值
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        # 如果周期数是5的倍数
        if (epoch) % 5 == 0:
        # if (epoch <= 100 and epoch % 5 == 0) or (epoch > 100):
            # 保存模型的状态字典到指定路径
            torch.save(model.state_dict(), save_path + 'PRNet_epoch_{}.pth'.format(epoch))

    # 如果程序被键盘中断（例如用户按下Ctrl+C）
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')

        # 如果保存路径不存在，则创建它
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 保存当前周期（或下一个周期）的模型状态字典到指定路径
        torch.save(model.state_dict(), save_path + 'PRNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')

        # 重新抛出键盘中断异常，使程序退出
        raise



# test function
def test(test_loader, model, epoch, save_path):
    # 定义测试函数，用于评估模型在测试集上的性能

    global best_mae, best_epoch
    # 声明全局变量best_mae和best_epoch，用于存储目前为止的最佳平均绝对误差(MAE)和对应的训练轮数

    model.eval()
    # 将模型设置为评估模式，关闭dropout和batch normalization等层的训练模式特性

    with torch.no_grad():
        # 使用torch.no_grad()上下文管理器，确保在评估模型时不会计算梯度，节省计算资源

        mae_sum = 0
        # 初始化MAE的累加和

        for i in range(test_loader.size):
            # 遍历测试数据加载器的所有批次

            image, gt, name, img_for_post = test_loader.load_data()
            # 从测试数据加载器中加载一个批次的数据，包括图像、真实标签、名称和用于后处理的图像

            gt = np.asarray(gt, np.float32)
            # 将真实标签转换为numpy数组，并指定数据类型为float32

            gt /= (gt.max() + 1e-8)
            # 将真实标签归一化到0到1之间，以避免除以零的错误

            image = image.cuda(1)
            # 将图像数据移动到GPU上进行计算（假设有可用的GPU）

            pred= model(image)
            # 将图像输入模型，获取模型的多个输出（可能是模型的不同层或分支的输出）

            # 将模型的多个输出相加，得到最终的预测结果

            res = F.upsample(pred, size=gt.shape, mode='bilinear', align_corners=False)
            # 使用双线性插值将预测结果上采样到与真实标签相同的尺寸

            res = res.sigmoid().data.cpu().numpy().squeeze()
            # 对预测结果应用sigmoid激活函数，将其转换为概率形式，然后移动到CPU上，转换为numpy数组，并去除单维度

            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # 对预测结果进行归一化，将其值范围调整到0到1之间

            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
            # 计算当前批次预测结果与真实标签之间的平均绝对误差(MAE)，并累加到mae_sum中

        mae = mae_sum / test_loader.size
        # 计算整个测试集的平均绝对误差(MAE)

        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        # 使用writer（可能是TensorBoard的SummaryWriter）将MAE记录为scalar类型的日志，以便可视化

        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        # 打印当前轮数的MAE、最佳MAE以及对应的最佳轮数

        if epoch == 1:
            # 如果是第一轮测试

            best_mae = mae
            # 将当前MAE设置为最佳MAE

        else:
            # 否则，进行最佳MAE的比较

            if mae < best_mae:
                # 如果当前MAE小于最佳MAE

                best_mae = mae
                # 更新最佳MAE

                best_epoch = epoch
                # 更新最佳MAE对应的轮数

                torch.save(model.state_dict(), save_path + 'best.pth')
                # 保存当前最佳模型的权重到指定路径

                print('best epoch:{}'.format(epoch))
                # 打印出最佳轮数

        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))
        # 使用logging模块记录测试阶段的日志信息


if __name__ == '__main__':
    # 检查当前脚本是否作为主程序运行，如果是，则执行以下代码块

    print("Start train...")
    # 打印开始训练的提示信息

    for epoch in range(1, opt.epoch):
        # 循环遍历训练轮数，从1开始，直到指定的训练轮数opt.epoch（不包括opt.epoch本身）

        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        # 调用adjust_lr函数，根据当前的训练轮数epoch调整学习率，并返回当前的学习率cur_lr

        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        # 使用writer（可能是TensorBoard的SummaryWriter）记录当前学习率作为scalar类型的日志，方便后续可视化

        train(train_loader, model, optimizer, epoch, save_path)
        # 调用train函数，使用训练数据加载器train_loader对模型进行训练，传入模型、优化器、当前训练轮数和模型权重保存路径

        test(test_loader, model, epoch, save_path)
