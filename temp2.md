分类网络 resnet18，resnet50，mobilenetv2，repvgg，mobileone，poolformer，CaiT，Deit，EfficientFormer，EfficientNet，MnasNet,MobileNets,MobileOne,MobileViT,MogaNet,PoolFormer,PVT/PVT-v2,RepVGG-A0,RepViT,ResNet SwinTransformer，T2T-vit，TinyViT，ViT，xcit ptcv_mobilenetv3_large_w1,densenet_121,regnet_x_32gf,mobilenetv4_conv_small,mobilenetv4_hybrid_medium,swin_tiny_patch4_window7_224,swinv2_base_patch4_window8_256 检测网络 yolo,ViDT,Sparse DETR，SAM-DETR，RT-DETR，RF-DETR，MIMDet,GroundingDINO，DN-DETR，DINO，DETR，Deformable DETR，DAB-DETR dn_detr_resnet50,conditional_detr_resnet50,detr_resnet50 分割网络 MaskFormer，SegFormer，topformer，upernet_deit,Segmenter ann_r50,dnl_r50,fcn_r50,danet_r50,fcn_hrnet,pspnet_mobilenetv2,pspnet_r50





网络模型熟悉

onnxruntime-gpu

因为公司不能使用conda，所以只能用pycharm进行虚拟环境的管理，相关的操作有哪些❌

docker相关命令✅

docker build -t my-app:1.0 . 这个命令是构建一个my-app:1.0的container？\# -p 8080:80: 将主机的8080端口映射到容器的80端口 这个意思是我通过主机8080端口找到的是容器的80端口吗，容器的80端口会被一直占用吗？✅

MAC利用率排查的相关Linux命令有哪些，是什么原理

给我讲解明白pytorch中的各种hook函数的开发应用原理。

服务器上默认安装的python的解释器是在哪个路径中找的，如果我想将默认打开的解释器版本更改为我自己重新安装的新版本如何实现 

0时刻拉取了某一工程，只对某一子文件夹做了修改。然后这时候再重新拉取这个工程，刚才做修改的文件夹会被覆盖吗，修改的内容会保存吗，还是需要git add .才会保存否则就会丢失。 

给我讲解明白Gerrit中不同的download方式区别

 Branch方式 git fetch ... && git checkout -b change-532087 FETCH_HEAD 

Checkout方式 git fetch ... && git checkout FETCH_HEAD 

Cherry Pick方式  git fetch ... && git cherry-pick FETCH_HEAD 

Format Patch方式 git fetch ... && git format-patch -1 --stdout FETCH_HEAD 

Pull方式  git pull ... 

Reset To方式 git fetch ... && git reset --hard FETCH_HEAD git pull origin master 

查看终端历史命令或者最近多少条命令的相关指令是什么✅

我现在想快速了解网络中的infernece pipeline，我有手动实现过DAG计算图的开发经验，但希望能快速了解一个新提出的网络结构。我有onnx的模型上面有详细的数据维度标注。如果网络提出了一些自定义的结构，我应该如何快速通过onnx熟悉网络执行流程。✅



ViT_b_16_224.onnx的模型包含以下算子，给我讲解这些算子的数据维度变化Gather(indices=1),Gather(indices=0),Unsquenze(axes<1>),Slice(start<1>,ends<1>,axes<1>),Transpose,ConstantOfShape,Mul(B=-1),Equal,Where,Expand,Add(B),ReduceMean,Sub,Pow(Y=2),Shape,MatMul,Erf,Div,Transpose,Gemm,Reshape(shape<1>),Concat,Sqrt,Div,Mul(B<768>)







矩阵论的哪些基本原理我需要知道，给我满足深度学习计算最基础的原理就行不需要全面复杂的。对应pytorch哪些函数，比如concat函数。✅

给我讲解明白python中运行时的各种路径关系，比如什么时候使用绝对路径，相对路径的判断范围，包的导入和init.py的关系等，要尽量全面详细。

python中高级语法

神经网络训练调参中问题汇总。比如，loss中震荡或不下降。

详细介绍列举一下当前data free剪枝方法，以及原理。

除了data free还有哪些不用大规模微调的稀疏化方法，是什么原理。

最新的剪枝还是选特定层吗，就是预先筛选出需要剪枝的层进行限制保证不会剪枝到其他的层。

我们办公设备是采用的client设备，我又一个外网设备有个内网设备，给我讲解一下我在连接服务器时经常看到的隧道的原理✅

 英伟达提出的2:4的N:M稀疏加速单元是应用于训练场景的吗？如果是做推理场景的这种2:4实现的必要性如何，我了解到还有一种是零矩阵加速的，就是针对1x2或者1x4的全零矩阵的稀疏化有加速效果，这种现在有实现的吗✅

我还是不理解为什么2:4能提升效果，硬件进行矩阵乘法具体是如何操作的，在4个内存块内随机2个不为0这样不还是不规则稀疏的吗？除此之外我还读到过某个设备能实现4倍提升，这个最有可能指什么的4倍提升？✅

稀疏化后，能在算法层面进行稀疏的重排吗，因为大多数非结构剪枝算法都是不规则稀疏的，如果把这些全部重排保证参数内存排列的规则化，这种实现有什么理论上的问题吗✅

我现在入职半导体公司，用最专业的话先给我讲明白ASIC和其他类型芯片都是什么，有什么区别。2018年四款AI芯片在行车记录仪与USB摄像头芯片领域市占率居行业首位 [3]。2019年发布轩辕、越影、降龙三大系列芯片并被认定为国家高新技术企业 [2] [5]。其IPC SoC和NVR SoC全球市场份额达36.5%、38.7%，USB视频会议摄像头芯片全球市占率51.8% [11]。以上提到的IPC SoC、NVR SoC和USB视频会议摄像头芯片分别都什么意思✅

给我讲讲AI芯片中，全面细致的讲讲为什么非规则稀疏不能被有效加速。✅

给我讲讲AI芯片中，推理场景下和并行运算加速有关的相关概念和原理知识。✅

我在训练时，显卡打印35.12s/it，这个数值越大表明显卡性能越差吗✅



1.空洞卷积，深度可分离卷积，分组卷积

(192,8,3,3)中的[:;:;0;0]到底代表什么含义



通过摘要用中文给我讲解文章的贡献







conda打开sigmastar的虚拟环境后，安装一些深度学习依赖，我的python版本是3.8，需要安装的torch版本是2.3.1，onnx的版本是1.15.0







我现在想对我的python文件通过调用argparse实现终端信息的输入，但现在存在问题，比如我希望通过终端传入模型的名称比如onnx_tv_resnet18，然后我希望程序中执行from onnx_tv_resnet18.py import onnx_preprocess，能实现吗用python代码简单举例。另外为什么有时候传参是双横线--，有时是单横线-





我现在需要实现一个脚本all.sh。

我当前文件夹结构是：

 ~/Synchronize/onnx_tv_resnet18/ tree

**.**

├── onnx_tv_resnet18.onnx

├── onnx_tv_resnet18.py

└── **run_sh**

  ├── 1.sh

  ├── 2.sh

  ├── 3.sh

  ├── 4.sh

  ├── all.sh

  ├── get_model.py

  ├── **out_put**

  └── run_dataset.py

该脚本现在已经实现从绝对路径中脚本当前路径run_sh的上一级地址parent_dir，取得模型名称model_name，此处为onnx_tv_resnet18

现在需要完善该脚本实现以下功能：

1.把上一级路径parent_dir中的onnx_tv_resnet18.py，即${model_name}.py放入run_sh中

2.执行当前路径中的get_model.py，即python3 get_model.py --input model ${model_name}

3.等待上一步执行完成后，执行当前路径中的run_dataset.py，即python3 run_dataset.py --input model ${model_name}

4.等待上一步执行完成后执行一个命令行的python3（对应原来1.sh的内容，我现在直接全部复制到all.sh中），这里不需要关注，随便写一些内容做占位符即可

5.从out_put文件夹中取出一个txt文件放入当前路径run_sh下，这个txt文件是out_put中唯一的txt文件，并且文件夹的命名中包含模型名称${model_name}

6.删除run_sh中的onnx_tv_resnet18.py