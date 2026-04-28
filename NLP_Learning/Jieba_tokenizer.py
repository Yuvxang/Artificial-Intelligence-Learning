import jieba
from jieba import lcut_for_search

content = "你说的对，但是《原神》是由米哈游自主研发的一款全新开放世界冒险游戏。游戏发生在一个被称作提瓦特的幻想世界，在这里，被神选中的人将被授予神之眼"
# 推荐 精确模式
def all_mode():
    words = jieba.lcut(content)
    print(words)

    # 不带l的返回generator,可以转化为list使用，效果和lcut一样
    result = jieba.cut(content)
    print(result)

# 全词模式
def precise():
    words = jieba.lcut(content, cut_all=True)
    print(words)

# 搜索引擎模式
def search():
    result = lcut_for_search(content)
    print(result)

if __name__ == "__main__":
    # 张量中不能放非数字
    search()