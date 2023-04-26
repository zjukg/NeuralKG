### [CCKS2023 开放环境下知识图谱构建与补全评测任务二：归纳式知识图谱关系推理](https://tianchi.aliyun.com/competition/entrance/532081/introduction?spm=5176.12281949.0.0.605a3b746tzJxF)

#### 如何运行

+ 进入Neuralkg文件夹（之后的相对位置都相对此目录）

```shell
cd NeuralKG
```

+ 切换到ind分支

```shell
git checkout ind
```

+ 在main.py最上方添加以下代码，使本地的neuralkg-ind目录优先于其他目录被import检查

```shell
import sys
sys.path.insert(0,"./src")
```

+ 运行.sh文件

```shell
sh ./scripts/CCKS/Grail.sh
```

运行结束后，模型在`./output`目录下，`./userFile.json`就是结果文件，key是query的index，value为对应query的候选尾实体排序，越有可能是真实的尾实体，排序越靠前。

#### 数据集

数据集分别为CCKS_train和CCKS_train_ind，这两个数据集都要放在`./dateset`下，训练集的文件夹名称可以按照个人喜好修改，但是测试集的文件夹必须为训练集+"_ind"，两个文件夹下面的文件名都要和示例中一样，不能修改。

`CCKS_train/train.txt`是训练集中的支持集，`CCKS_train/valid.txt`是验证集，您可以选择不用验证集，跑固定的epoch或step，或者将比赛中给的训练集拆分成支持集和验证集。示例的方法是不用验证集，跑固定的epoch，但是为了对标inductive模型，`CCKS_train/valid.txt`不能为空，你可以随便将`CCKS_train/train.txt`里面的一个triple复制过来，因为这个对结果没有任何影响（我们只关注模型在测试集上的效果）。

CCKS_train_ind下面包含支持集train.txt和查询集test_query.json，其中train.txt是比赛中test_support.txt。

#### .sh文件

我们在`scripts/CCKS/Grail.sh`中给出了使用GraIL模型进行训练的相关参数，使用的数据集为CCKS_train，`max_epochs`为7，也就是只在训练集上跑7个epoch，因为最后需要保存模型所以`check_per_epoch`要与`max_epochs`一致，另外还有`--ccks`是为了使用CCKS比赛中的一些配置，其他的参数都与标准的GraIL模型一致。

#### 源码中修改的部分

+ 在`src/neuralkg_ind/data/DataPreprocess/BaseGraph/generate_ind_test`方法中，获得neg_triplets的时候我们不是随机地破坏头尾实体，而是读入test_query.json文件，根据候选尾实体构建负样本。

+ 在`src/neuralkg_ind/data/Sampler.py/TestSampler_hit/sampling`方法中，我们将所有候选尾实体保存在`batch_data['tails']`中。

+ 在`src/neuralkg_ind/eval_task/link_prediction/ind_predict`方法中，根据模型返回的score对候选尾实体进行排序。
+ 在`src/neuralkg_ind/lit_model/BaseLitModel.py/BaseLitModel/get_results`方法中，将排序号的候选尾实体保存为userFile.json。
+ 添加超参数`--ccks`，用于控制上述修改生效是否生效。



