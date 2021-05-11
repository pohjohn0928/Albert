from pathlib import Path
import csv
from sklearn.utils import shuffle
import cchardet

class DataHelper:
    def __init__(self):
        self.root = Path().cwd()

    def get_data(self,file_name):
        file_path = self.root / file_name

        file = open(file_path,'rb')
        encoding = cchardet.detect(file.read())['encoding']

        # print(encoding)
        csvfile = open(file_path, newline='',encoding=encoding)
        reader = csv.DictReader(csvfile)
        contents = []
        labels = []
        for row in reader:
            contents.append(row.get('content',''))
            labels.append(row.get('label',''))
        contents,labels = shuffle(contents,labels)
        return contents,labels