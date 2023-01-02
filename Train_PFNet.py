import os
import sys
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils
from utils import PointLoss
from utils import distance_squre
import data_utils as d_utils
import ModelNet40Loader
import shapenet_part_loader
from model_PFNet import _netlocalD,_netG
import point_dump as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',  default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int,default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=512,help='0 means do not use else use with this weight')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--D_choose',type=int, default=1, help='0 not use D-net,1 use D-net')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--point_scales_list',type=list,default=[2048,1024,512],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.95,help='0 means do not use else use with this weight')
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')
opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USE_CUDA = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
point_netG = _netG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num)  # (3,1,[2048,1024,512],512)
point_netD = _netlocalD(opt.crop_point_num)  # (512)
cudnn.benchmark = True  # 增加运行效率 自动寻找最适合的卷积算法
resume_epoch=0

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0) 

if USE_CUDA:       
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    point_netG = torch.nn.DataParallel(point_netG)
    point_netD = torch.nn.DataParallel(point_netD)
    point_netG.to(device) 
    point_netG.apply(weights_init_normal)
    point_netD.to(device)
    point_netD.apply(weights_init_normal)
if opt.netG != '' :
    point_netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']
if opt.netD != '' :
    point_netD.load_state_dict(torch.load(opt.netD,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD)['epoch']

        
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)


transforms = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
    ]
)
dset = shapenet_part_loader.PartDataset( root='./dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice=None, npoints=opt.pnum, split='train')
assert dset
dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,
                                         shuffle=True,num_workers = int(opt.workers))

test_dset = shapenet_part_loader.PartDataset( root='./dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice=None, npoints=opt.pnum, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                         shuffle=False,num_workers = int(opt.workers))

#dset = ModelNet40Loader.ModelNet40Cls(opt.pnum, train=True, transforms=transforms, download = False)
#assert dset
#dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,
#                                         shuffle=True,num_workers = int(opt.workers))
#
#
#test_dset = ModelNet40Loader.ModelNet40Cls(opt.pnum, train=False, transforms=transforms, download = False)
#test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
#                                         shuffle=True,num_workers = int(opt.workers))

#pointcls_net.apply(weights_init)
print(point_netG)
print(point_netD)

# 损失函数
criterion = torch.nn.BCEWithLogitsLoss().to(device)
criterion_PointLoss = PointLoss().to(device)

# setup optimizer 优化器
optimizerD = torch.optim.Adam(point_netD.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05,weight_decay=opt.weight_decay)
optimizerG = torch.optim.Adam(point_netG.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05 ,weight_decay=opt.weight_decay)
schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=40, gamma=0.2)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.2)

real_label = 1
fake_label = 0

crop_point_num = int(opt.crop_point_num)
input_cropped1 = torch.FloatTensor(opt.batchSize, opt.pnum, 3)
label = torch.FloatTensor(opt.batchSize)


num_batch = len(dset) / opt.batchSize
###########################
#  G-NET and T-NET
##########################  
if opt.D_choose == 1:
    for epoch in range(resume_epoch,opt.niter):
        if epoch<30:
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch<80:
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            alpha1 = 0.1
            alpha2 = 0.2

        # 原始输入点云：2048个点 丢失512个点 batch：8
        for i, data in enumerate(dataloader, 0):
            
            real_point, target = data  # real_point(8,2048,3) target(8,1)

            real_center, real_point, input_cropped1 = pd.dump_point(opt.pnum, opt.crop_point_num, real_point, opt.cropmethod)

            batch_size = real_point.size()[0]  # 8
            # real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)  # (8,1,512,3) 需要丢弃的点的总数
            # print(real_center.shape)
            # input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)  # (8,2048,3) 开始输入的点云
            # print(input_cropped1.shape)
            # input_cropped1 = input_cropped1.data.copy_(real_point)  # (8,1,2048,3)
            # print(input_cropped1.shape)
            # real_point = torch.unsqueeze(real_point, 1)  # (8,1,2048,3)
            # print(real_point.shape)
            # input_cropped1 = torch.unsqueeze(input_cropped1,1)  # (8,1,2048,3)
            # print(input_cropped1.shape)
            # p_origin = [0,0,0]
            # # 丢弃点
            # if opt.cropmethod == 'random_center':
            #     #Set viewpoints
            #     choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]  # 指定丢弃时可选择的初始位置（角落）
            #     for m in range(batch_size):
            #         # 随机选择一个位置
            #         index = random.sample(choice,1)#Random choose one of the viewpoint
            #         distance_list = []
            #         p_center = index[0]
            #         for n in range(opt.pnum):
            #             distance_list.append(distance_squre(real_point[m,0,n],p_center))  # real_point[m,0,n]每个点都要遍历 p_center每个点到center的距离
            #         distance_order = sorted(enumerate(distance_list), key  = lambda x:x[1])  # 按距离远近排序
            #         print(distance_order)
            #         for sp in range(opt.crop_point_num):  # 把一部分点去掉，并且保存其标签  构造每一个缺失点 512
            #             input_cropped1.data[m,0,distance_order[sp][0]] = torch.FloatTensor([0,0,0])  # 丢掉点
            #             real_center.data[m,0,sp] = real_point[m,0,distance_order[sp][0]]  # 保存点的实际值
            label.resize_([batch_size,1]).fill_(real_label)  # (8,1))
            print(label.shape)
            real_point = real_point.to(device)
            real_center = real_center.to(device)
            input_cropped1 = input_cropped1.to(device)
            label = label.to(device)
            ############################
            # (1) data prepare  准备数据 512，1024，2048 拼接
            ###########################
            # 丢失点的实际值 Ground Truth
            real_center = Variable(real_center,requires_grad=True)  # (8,1,512,3) Variable()封装了Tensor，并整合了反向传播的相关实现
            real_center = torch.squeeze(real_center,1)  # (8,512,3)
            # 采样第一个点云：64 实际值
            real_center_key1_idx = utils.farthest_point_sample(real_center,64,RAN = False)  # (8,64) 最远点采样
            real_center_key1 = utils.index_points(real_center,real_center_key1_idx)  # (8,64,3) 按输入的点云数据和索引返回由索引的点云数据
            real_center_key1 =Variable(real_center_key1,requires_grad=True)  # (8,64,3)
            print(real_center_key1.shape)
            # 采样第二个点云：128 实际值
            real_center_key2_idx = utils.farthest_point_sample(real_center,128,RAN = True)  # (8,128)
            real_center_key2 = utils.index_points(real_center,real_center_key2_idx)  # (8,128,3)
            real_center_key2 =Variable(real_center_key2,requires_grad=True)  # (8,128,3)
            print(real_center_key2.shape)

            # 原始的已存在缺失部分的点云：2048（第一个点云） 点的生成值
            input_cropped1 = torch.squeeze(input_cropped1,1)  # (8,2048,3) 维度压缩，去掉维数为1的维度
            print(input_cropped1.shape)
            # 采样第二个点云：1024
            input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)  # (8,1024)
            input_cropped2 = utils.index_points(input_cropped1,input_cropped2_idx)  # (8,1024,3)
            #采样第三个点云：512
            input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)  # (8,512)
            input_cropped3 = utils.index_points(input_cropped1,input_cropped3_idx)  # (8,512,3)
            input_cropped1 = Variable(input_cropped1,requires_grad=True)  # (8,2048,3)
            print(input_cropped1.shape)
            input_cropped2 = Variable(input_cropped2,requires_grad=True)  # (8,1024,3)
            print(input_cropped2.shape)
            input_cropped3 = Variable(input_cropped3,requires_grad=True)  # (8,512,3)
            print(input_cropped3.shape)
            input_cropped2 = input_cropped2.to(device)
            input_cropped3 = input_cropped3.to(device)
            # 将采样的三个点云拼接起来
            input_cropped  = [input_cropped1,input_cropped2,input_cropped3]  # 拼接

            # 准备开始训练模型
            point_netG = point_netG.train()
            point_netD = point_netD.train()
            ############################
            # (2) Update D network 训练判别器
            ###########################        
            point_netD.zero_grad()  # 梯度清零
            real_center = torch.unsqueeze(real_center,1)  # (8,1,512,3) 增加数组指定位置的维度
            output = point_netD(real_center)  # (8,1) 使用判别器对一个批次真的进行分类
            errD_real = criterion(output,label)  # 计算判断真实的损失值
            errD_real.backward()
            fake_center1, fake_center2, fake = point_netG(input_cropped)  # 生成器生成点 fake_center1(8,64,3) fake_center2(8,128,3) fake(8,512,3)
            fake = torch.unsqueeze(fake,1)  # (8,1,512,3) 增加数组fake指定位置的维度
            label.data.fill_(fake_label)  # 标签全部改为0
            output = point_netD(fake.detach())  # (8,1) 使用判别器对一个批次假的进行分类
            errD_fake = criterion(output, label)  # 计算判断为假的损失值
            errD_fake.backward()
            errD = errD_real + errD_fake  # 判断真和判断假的损失值加和作为总损失
            optimizerD.step()
            ############################
            # (3) Update G network: maximize log(D(G(z))) 训练生成器
            ###########################
            point_netG.zero_grad()
            label.data.fill_(real_label)  # 标签全部改为1
            output = point_netD(fake)  # (8,1) 判别刚才生成的假的
            errG_D = criterion(output, label)  # 计算判断假的为真的损失值
            errG_l2 = 0
            CD_LOSS = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))  # 计算生成器生成点和原始点的损失

            # 总损失 生成器每层生成的点与原始每层的点
            errG_l2 = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))\
            +alpha1*criterion_PointLoss(fake_center1,real_center_key1)\
            +alpha2*criterion_PointLoss(fake_center2,real_center_key2)

            errG = (1-opt.wtl2) * errG_D + opt.wtl2 * errG_l2
            errG.backward()
            optimizerG.step()
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f/ %.4f'
                  % (epoch, opt.niter, i, len(dataloader), 
                     errD.data, errG_D.data,errG_l2,errG,CD_LOSS))  # 输出打印格式
            f=open('loss_PFNet.txt','a')  # 保存loss
            f.write('\n'+'[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f /%.4f'
                  % (epoch, opt.niter, i, len(dataloader), 
                     errD.data, errG_D.data,errG_l2,errG,CD_LOSS))


            if i % 10 ==0:
                print('After, ',i,'-th batch')
                f.write('\n'+'After, '+str(i)+'-th batch')
                for i, data in enumerate(test_dataloader, 0):
                    real_point, target = data  # real_point(8,2048,3) target(8,1)
                    
            
                    batch_size = real_point.size()[0]  # 8
                    real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)  # (8,1,512,3)
                    input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)  # (8,2048,3)
                    input_cropped1 = input_cropped1.data.copy_(real_point)  # (8,2048,3)
                    real_point = torch.unsqueeze(real_point, 1)  # (8,1,2048,3)
                    input_cropped1 = torch.unsqueeze(input_cropped1,1)  # (8,1,2048,3)
                    
                    p_origin = [0,0,0]
                    
                    if opt.cropmethod == 'random_center':
                        choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
                        
                        for m in range(batch_size):
                            index = random.sample(choice,1)
                            distance_list = []
                            p_center = index[0]
                            for n in range(opt.pnum):
                                distance_list.append(distance_squre(real_point[m,0,n],p_center))
                            distance_order = sorted(enumerate(distance_list), key  = lambda x:x[1])                         
                            for sp in range(opt.crop_point_num):
                                input_cropped1.data[m,0,distance_order[sp][0]] = torch.FloatTensor([0,0,0])  # (8,1,2048,3)
                                real_center.data[m,0,sp] = real_point[m,0,distance_order[sp][0]]  # (8,1,512,3)
                    real_center = real_center.to(device)
                    real_center = torch.squeeze(real_center,1)  # (8,512,3)
                    input_cropped1 = input_cropped1.to(device) 
                    input_cropped1 = torch.squeeze(input_cropped1,1)  # (8,2048,3)
                    input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)  # (8,1024)
                    input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)  # (8,1024,3)
                    input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)  # (8,512)
                    input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)  # (8,512,3)
                    input_cropped1 = Variable(input_cropped1,requires_grad=False)  # (8,2048,3)
                    input_cropped2 = Variable(input_cropped2,requires_grad=False)  # (8,1024,3)
                    input_cropped3 = Variable(input_cropped3,requires_grad=False)  # (8,512,3)
                    input_cropped2 = input_cropped2.to(device)
                    input_cropped3 = input_cropped3.to(device)      
                    input_cropped  = [input_cropped1,input_cropped2,input_cropped3]  # 拼接
                    point_netG.eval()
                    fake_center1,fake_center2,fake  =point_netG(input_cropped)  # fake_center1(8,64,3) fake_center2(8,128,3) fake(8,512,3)
                    CD_loss = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))
                    print('test result:',CD_loss)
                    f.write('\n'+'test result:  %.4f'%(CD_loss))
                    break
            f.close()
        schedulerD.step()
        schedulerG.step()
        if epoch% 10 == 0:   # 保存模型
            torch.save({'epoch':epoch+1,
                        'state_dict':point_netG.state_dict()},
                        'Trained_Model/point_netG/point_netG'+str(epoch)+'.pth' )
            torch.save({'epoch':epoch+1,
                        'state_dict':point_netD.state_dict()},
                        'Trained_Model/point_netD/point_netD'+str(epoch)+'.pth' )

#
#############################
## ONLY G-NET
############################ 
else:
    for epoch in range(resume_epoch,opt.niter):
        if epoch<30:
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch<80:
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            alpha1 = 0.1
            alpha2 = 0.2
        
        for i, data in enumerate(dataloader, 0):
            
            real_point, target = data  # real_point(8,2048,3) target(8,1)
            
    
            batch_size = real_point.size()[0]  # 8
            real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)  # (8,1,512,3)
            input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)  # (8,2048,3)
            input_cropped1 = input_cropped1.data.copy_(real_point)  # (8,2048,3)
            real_point = torch.unsqueeze(real_point, 1)  # (8,1,2048,3)
            input_cropped1 = torch.unsqueeze(input_cropped1,1)  # (8,1,2048,3)
            p_origin = [0,0,0]
            if opt.cropmethod == 'random_center':
                choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
                for m in range(batch_size):
                    index = random.sample(choice,1)
                    distance_list = []
                    p_center = index[0]
                    for n in range(opt.pnum):
                        distance_list.append(distance_squre(real_point[m,0,n],p_center))
                    distance_order = sorted(enumerate(distance_list), key  = lambda x:x[1])
                    
                    for sp in range(opt.crop_point_num):
                        input_cropped1.data[m,0,distance_order[sp][0]] = torch.FloatTensor([0,0,0])  # (8,1,2048,3)
                        real_center.data[m,0,sp] = real_point[m,0,distance_order[sp][0]]  # (8,1,512,3)
            real_point = real_point.to(device)
            real_center = real_center.to(device)
            input_cropped1 = input_cropped1.to(device)
            ############################
            # (1) data prepare
            ###########################      
            real_center = Variable(real_center,requires_grad=True)  # (8,1,512,3)
            real_center = torch.squeeze(real_center,1)  # (8,512,3)
            real_center_key1_idx = utils.farthest_point_sample(real_center,64,RAN = False)  # (8,64)
            real_center_key1 = utils.index_points(real_center,real_center_key1_idx)  # (8,64,3)
            real_center_key1 =Variable(real_center_key1,requires_grad=True)  # (8,64,3)

            real_center_key2_idx = utils.farthest_point_sample(real_center,128,RAN = True)  # (8,128)
            real_center_key2 = utils.index_points(real_center,real_center_key2_idx)  # (8,128,3)
            real_center_key2 =Variable(real_center_key2,requires_grad=True)  # (8,128,3)
            
            input_cropped1 = torch.squeeze(input_cropped1,1)  # (8,2048,3)
            input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)  # (8,1024)
            input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)  # (8,1024,3)
            input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)  # (8,512)
            input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)  # (8,512,3)
            input_cropped1 = Variable(input_cropped1,requires_grad=True)  # (8,2048,3)
            input_cropped2 = Variable(input_cropped2,requires_grad=True)  # (8,1024,3)
            input_cropped3 = Variable(input_cropped3,requires_grad=True)  # (8,512,3)
            input_cropped2 = input_cropped2.to(device)
            input_cropped3 = input_cropped3.to(device)      
            input_cropped  = [input_cropped1,input_cropped2,input_cropped3]  # 拼接
            point_netG = point_netG.train()
            point_netG.zero_grad()
            fake_center1,fake_center2,fake  =point_netG(input_cropped)  # fake_center1(8,64,3) fake_center2(8,128,3) fake(8,512,3)
            fake = torch.unsqueeze(fake,1)  # (8,1,512,3)
            ############################
            # (3) Update G network: maximize log(D(G(z)))
            ###########################
            
            CD_LOSS = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))
            
            errG_l2 = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))\
            +alpha1*criterion_PointLoss(fake_center1,real_center_key1)\
            +alpha2*criterion_PointLoss(fake_center2,real_center_key2)

            errG_l2.backward()
            optimizerG.step()
            print('[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
                  % (epoch, opt.niter, i, len(dataloader), 
                      errG_l2,CD_LOSS))
            f=open('loss_PFNet.txt','a')
            f.write('\n'+'[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
                  % (epoch, opt.niter, i, len(dataloader), 
                      errG_l2,CD_LOSS))
            f.close()
        schedulerD.step()
        schedulerG.step()
        
        if epoch% 10 == 0:
            torch.save({'epoch':epoch+1,
                        'state_dict':point_netG.state_dict()},
                        'Checkpoint/point_netG'+str(epoch)+'.pth' )
 

    
        
