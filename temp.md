

>conda打开sigmastar的虚拟环境后，安装一些深度学习依赖，我的python版本是3.10，需要安装的torch版本是，onnx的版本是



我现在想对我的python文件通过调用argparse实现终端信息的输入，但现在存在问题，比如我希望通过终端传入模型的名称比如onnx_tv_resnet18，然后我希望程序中执行from onnx_tv_resnet18.py import onnx_preprocess，能实现吗用python代码简单举例。另外为什么有时候传参是双横线--，有时是单横线-



pytorch有没有现成能提供参数化剪枝的库，我当前路径下只有一个resnet18.onnx，提供python程序，对该onnx文件执行参数剪枝





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



