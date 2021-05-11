## Tesnsorflow : feature base
- 用 bert pretrain model 完成特徵轉換,直接將特徵放入自己的模型進行訓練(不會更新pretrain model的參數)
- 用 Fine Tunning : 完成特徵轉換,將特徵跟pretrain model的權重放入訓練(會更新pretrain model的參數)


- Bert : hugging face and transformer pretrain model
- use tensorflow


## TensorBoard
- tensorboard --logdir logs/