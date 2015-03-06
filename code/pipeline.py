from process_data import ProcessData
from classify import ClassifyRep
import pandas as pd

if __name__ == '__main__':
	info = pd.read_csv('../data/pushup_info.csv', skipinitialspace=True)
	p = ProcessData(info,'narrow',plot=False)
	p.batch_process_samples()

	c = ClassifyRep(X, labels)
	sss = c.split_data()
	c.random_forest(sss)
	