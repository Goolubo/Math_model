# 石油

目标：

- 通过数据挖掘技术来建立汽油辛烷值（RON）损失的预测模型，并给出每个样本的优化操作条件
- 本次建模要求产品硫含量不大于5μg/g，尽量降低汽油辛烷值损失在30%以上。

任务：

1. 数据处理

   1. 附件1是预处理结果

      1. 可以认为辛烷值的测量值是测量时刻前两小时内操作变量的综合效果，因此预处理中取操作变量两小时内的平均值与辛烷值的测量值对应。这样产生了325个样本（见附件一）

   2. 附件三是285和313样本的原始数据

      1. 2017年4月至2019年9月，数据采集频次为3分钟/次；2019年10月至2020年5月，数据采集频次为6分钟/次。

   3. 依附件2对285和313样本进行预处理，并加入到附件一中（原始数据在附件三）

      1. 对于只含有部分时间点的位点，如果其残缺数据较多，无法补充，将此类位点（变量）删除；

      2. 删除325个样本中数据全部为空值的位点；

      3. 对于部分数据为空值的位点，空值处用其前后两个小时数据的平均值代替；

      4. **根据工艺要求与操作经验，总结出原始数据变量的操作范围，然后采用最大最小的限幅方法剔除一部分不在此范围的样本；**

      5. **根据拉依达准则（3σ准则）去除异常值。**

         > 3σ准则：设对被测量变量进行等精度测量，得到 $ x_1，x_2，……，x_n，$算出其算术平均值 $x$及剩余误差 $v_i=x_i-x（i=1,2,...,n）$，并按贝塞尔公式算出标准误差 $σ$，若某个测量值 $x_b$的剩余误差 $v_b（1\leqslant b\leqslant n）$，满足 $|v_b|=|x_b-x|>3σ$ ，则认为 $ x_b$是含有粗大误差值的坏值，应予剔除。贝塞尔公式如下：
         > $$\sigma=[\frac1{n-1}\sum_{i=1}^nv_i^2]^{1/2}=\{[\sum_{i=1}^nx_i^2-(\sum_{i=1}^nx_i)^2/n]/(n-1)\}^{1/2}$$


2. 寻找建模主要变量

   1. 先降维后建模
      1. ==7个原料性质、2个待生吸附剂性质、2个再生吸附剂性质、2个产品性质等变量以及另外354个操作变量（共计367个变量）==
      2. 主要变量在30个以下
   2. 考虑将原料的辛烷值作为建模变量之一

3. 建立辛烷值损失预测模型

4. 主要变量操作方案的优化

   1. 对325个数据样本（见附件一），取出其中所有按预测模型，产品硫含量不大于5μg/g，辛烷值（RON）损失降幅大于30%的样本
   2. 列表给出它们对应的主要变量优化后的操作条件（优化过程中原料、待生吸附剂、再生吸附剂的性质保持不变，以它们在样本中的数据为准）。

5. 模型可视化展示

   1. 对133号样本（原料性质、待生吸附剂和再生吸附剂的性质数据保持不变，以样本中的数据为准），以图形展示其主要操作变量优化调整过程中对应的汽油辛烷值和硫含量的变化轨迹。
   2. 各主要操作变量每次允许调整幅度值Δ见附件四