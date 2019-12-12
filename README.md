# LIP-TSM

## 基础信息

**本机环境**：（皆为最新版本）

- 单卡1080Ti
- ubuntu18.04
- CUDA10
- CUDNN

**安装：**

```
conda create -n paddlepaddle python=3.7
source activate paddlepaddle
cd Lip-TSM-1
pip install paddlepaddle-gpu
pip install requirements.txt
```

**数据集路径设置**：

解压压缩包，将lip_train、lip_test、lip_train.txt放置到一个data的目录下

```
data:
    /lip_train
    /lip_test
    /lip_train.txt
```

## 操作步骤

**第一步（数据准备，同Lip-TSM-1一样，如若已执行，无需再执行）：**	

​	修改 tools/make_dataset.py 文件__main__下的数据集路径data_dir为自己的数据集路径'/xxx/data/'，然后执行

```
python tools/make_dataset.py
```

最终会在数据集所在路径生成 train.list、val.list、test.list三个列表，和train、val、test三个文件夹

**第二步（配置文件）：**
	修改 configs/tsm.txt配置文件里的所有filelist文件路径信息
	总共[TRAIN]、[VALID]、[TEST]、[INFER]，四个路径需要修改的

**第三步（开始训练）：**

1、利用Lip-TSM-1的模型作为预训练，复制Lip-TSM-1目录下的checkpoints_models_best模型文件夹到Lip-TSM-2目录下，请使用--resume 权重路径，执行 

```
python train.py --epoch 200 --use_gpu True --resume checkpoints_models_best/
```
2、如若从头训练，执行 

```
python train.py --epoch 200 --use_gpu True --pretrain False
```
**第四步：**

预测结果会有以时间戳命名的两个文件

- 15xxxxxxxx_result.csv
- score_15xxxxxxxx_result.csv

第一个文件可以直接用来提交结果，第二个文件是用来进行模型融合的（包含了score列），执行

```
 python infer.py --weights checkpoints_models_best082
```

