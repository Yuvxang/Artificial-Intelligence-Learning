import fasttext

def model_preparation():
    model = fasttext.train_unsupervised("./gz03ag")
    model.save_model('word2vec.pkl')

def get_word_vec(word):
    model = fasttext.load_model("word2vec.pkl")
    vec = model.get_word_vector(word)
    print(vec)

def get_near_words(word):
    model = fasttext.load_model("word2vec.pkl")
    listword = model.get_nearest_neighbors(word)
    for item in listword:
        print(item[1])

def detailed_settings():
    model = fasttext.train_unsupervised(
        input="./gz03ag",
        model="cbow",
        dim=100,
        epoch=1,
        lr=0.01,
        threads=10
    )
    model.save_model("greatermodel.pkl")

if __name__ == '__main__':
    get_near_words("dog")