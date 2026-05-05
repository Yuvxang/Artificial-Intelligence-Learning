import fasttext

def first_trial():
    # 不需要定义优化器/损失函数等
    # 1.模型训练 有监督（回归/分类）、无监督（聚类）
    model = fasttext.train_supervised(input="./fasttext_data/cooking_train.txt")
    # 参数可以去源代码 - supervised_default 后面的 arg_names查看
    # 重要的主要有以下几个

    # input - 训练数据路径
    # lr - 学习率
    # epoch - 训练轮次
    # wordNgrams Ngram特征
    # autotuneValidationFile
    # autotuneDuration

    # 2.模型预测
    pred = model.predict("What are the names of the breakfast spreads used in Indian cuisine?", k=3)
    print(pred)

    # 3.模型评估
    test_result = model.test("./fasttext_data/cooking_valid.txt")
    print(test_result)
    # 结果解释：3000评估样本条数 0.129666 精确率 0.056076召回率

def predata_train():
    # 参数调整优化
    # 将字母统一写成小写，标点符号前增加空格
    model = fasttext.train_supervised(input="./fasttext_data/cooking.pre.train")
    result = model.test("./fasttext_data/cooking.pre.valid")
    print(result)

def epoch_addition():
    # 增加训练轮次且使用小写的
    model = fasttext.train_supervised(input="./fasttext_data/cooking.pre.train", epoch=20)
    result = model.test("./fasttext_data/cooking.pre.valid")
    print(result)

def learning_rate_adjustment():
    model = fasttext.train_supervised(input="./fasttext_data/cooking.pre.train", epoch=20, lr=1)
    result = model.test("./fasttext_data/cooking.pre.valid")
    print(result)

def NgramTrait():
    model = fasttext.train_supervised(input="./fasttext_data/cooking.pre.train", epoch=20, lr=1, wordNgrams=2)
    result = model.test("./fasttext_data/cooking.pre.valid")
    print(result)

def lossfunc():
    # 使用层次softmax函数为损失函数,训练速度提升log2
    # 精确率/召回率变化不大
    model = fasttext.train_supervised(input="./fasttext_data/cooking.pre.train", epoch=20, lr=1, wordNgrams=2, loss='hs')
    result = model.test("./fasttext_data/cooking.pre.valid")
    print(result)

def autotune_Settings():
    # 自动超参调优 duration是时间的上限(s),validationFile是效果验证文件
    model = fasttext.train_supervised(input="./fasttext_data/cooking.pre.train",
                                        autotuneDuration = 60*2,
                                        autotuneValidationFile="./fasttext_data/cooking.pre.valid")
    # 感觉和teacherforcing很像

    result = model.test("./fasttext_data/cooking.pre.valid")
    print(result)

# 将多标签多分类问题简化成单标签多分类的问题,每一种分类单独进行训练
def ovalossfunc():
    model = fasttext.train_supervised(input="./fasttext_data/cooking.pre.train", epoch=20, lr=0.1, wordNgrams=2, loss='ova')
    result = model.test("./fasttext_data/cooking.pre.valid")
    print(result)

def save_model():
    model = fasttext.train_supervised(input="./fasttext_data/cooking.pre.train", epoch=20, lr=0.1, wordNgrams=2, loss='ova')
    model.save_model("./fasttext_data/cooking_pred.pkl")

    model = fasttext.load_model("./fasttext_data/cooking_pred.pkl")
    pred = model.predict("What are the names of the breakfast spreads used in Indian cuisine?")
    print(pred)

def examine(model):
    model.get_nearest_neighbor()

# 文本分类比较清楚，类型比较少的话，可以直接使用fasttext
if __name__ == '__main__':
    save_model()
