from typing import Iterator
from torch.utils.data import IterableDataset, Dataset, DataLoader
from csv import DictReader
import time

class MappedDataset(Dataset) : 

    def __init__(self) : 
        super().__init__()
        self.fake_news_path = 'data/Fake.csv'
        self.true_news_path = 'data/True.csv'

        self.fake_news = [{"news":data["text"], "headline":data["title"],"valid":0} 
                          for data in DictReader(open(self.fake_news_path))]
        
        self.true_news = [{"news":data["text"], "headline":data["title"],"valid":1} 
                          for data in DictReader(open(self.true_news_path))]
        
        self.news_data = self.fake_news + self.true_news


    def __len__(self) : 
        return len(self.news_data)
    
    def __getitem__(self, index) -> tuple:
        return self.news_data[index]["news"] ,  self.news_data[index]["headline"] , self.news_data[index]["valid"]
    

class IteratedDataset(IterableDataset) : 

    def __init__(self) -> None:
        super().__init__()

        self.fake_news_path = 'data/Fake.csv'
        self.true_news_path = 'data/True.csv'

        self.fake_news = [{"news":data["text"], "headline":data["title"],"valid":0} 
                        for data in DictReader(open(self.fake_news_path))]
        
        self.true_news = [{"news":data["text"], "headline":data["title"],"valid":1} 
                        for data in DictReader(open(self.true_news_path))]
        
        self.news_data = self.fake_news + self.true_news

    def __iter__(self) -> Iterator:
        return super().__iter__()


    


mapped_dataset = MappedDataset()

cal_house_dataloader = DataLoader(dataset=mapped_dataset, # the dataset instance
                                  batch_size=64,             # automatic batching
                                  drop_last=False,            # drops the last incomplete batch in case the dataset size is not divisible by 64
                                  shuffle=True               # shuffles the dataset before every epoch
                                  )


print(len(mapped_dataset))

for i, batch in enumerate(cal_house_dataloader) : 

    print(i)
    # print(batch)
    time.sleep(0.5)

