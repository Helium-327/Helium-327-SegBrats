# SegBrats

基于BraTS数据集的分割方案


---

**训练流程**

- [ ] 准备数据集
- [ ] 修改参数
- [ ] 训练模型
- [ ] 评估模型
- [ ] 生成结果

**测试流程**

- [ ] 准备数据集
- [ ] 修改参数
- [ ] 测试模型
- [ ] 评估模型
- [ ] 生成结果


## Running

```bash
git clone https://github.com/Helium-327/BraTS_segmentation.git

cd BraTS_segmentation

pip install -r requirements.txt

```


## TRAIN

- 全量训练

```bash
    python BraTS_worflow/train.py --bs 2 --epochs 100 --lr 0.0005 --train_mode full

```

- 少量训练

```bash
    python BraTS_worflow/train.py --bs 2 --epochs 100 --lr 0.0005 --train_mode local --local_train_length <训练集长度> --local_val_length <验证集长度>
```
