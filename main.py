from model_helper import AlbertModel
from data_helper import DataHelper


if __name__ == '__main__':
    dataHelper = DataHelper()
    contents,labels = dataHelper.get_data('test1.csv')

    albert = AlbertModel()
    # pre = albert.predict('Fully_Albert',['護膚品','兄弟'])
    # print(pre)
    albert.fully(contents,labels)
    albert.rnn(contents,labels)

    # model_helper = ModelHelper()
    # model_helper.feature_base_fit(contents,labels)
    # model_helper.fine_tunning_fit(contents,labels)
    # model_helper.predict_fine_tuning(['很棒的服務,到貨也很快','價格便宜','品質很差'])