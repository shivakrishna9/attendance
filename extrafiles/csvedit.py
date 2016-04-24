import pandas as pd

# reader1 = pd.read_csv('../traintest/dev_urls.txt', comment='#', sep='\t')
# reader2 = pd.read_csv('../traintest/eval_urls.txt', comment='#', sep='\t')
reader3 = pd.read_csv('../traintest/faceindex.txt', comment='#', sep='\t')

# print reader1

# # reader1['face_id'] = reader1[['person', 'imagenum']].apply(lambda x: '_'.join(x), axis=1)
# reader1["face_id"] = '_'.join(reader1["person"]) +'_'+ reader1["imagenum"].map(str)

# print reader1

# # pd.concat([reader1, reader2, reader3], axis=0, join='outer')
with open('../traintest/downloads.txt', 'a') as f:
	for i in reader3.itertuples():
		print str(i[1])+','+i[2]
		f.write(str(i[1])+','+i[2]+'\n')
