import pandas as pd
import time
# reader1 = pd.read_csv('../traintest/dev_urls.txt', comment='#', sep='\t')
# reader2 = pd.read_csv('../traintest/eval_urls.txt', comment='#', sep='\t')
print 'extracting'
start = time.time()
reader3 = pd.read_csv('../traintest/training.txt', sep='\t')
print 'done in ..', time.time()-start

with open('../traintest/training2.txt', 'w') as f:
	print 'file opened'
	f.write('person'+'\timage'+'\tbbox'+'\n')
	count = 0
	person = []
	print 'reading persons '
	start = time.time()
	for i in reader3.itertuples():
		if i[1] not in person:
			person += [i[1]]
		# print person+'\t'+image+'\t'+bbox
		# f.write(person+'\t'+image+'\t'+bbox+'\n')
	print 'people read in ..', time.time()-start

	print 'writing to file'
	start = time.time()
	for i in reader3.itertuples():
		name = i[1]
		image = i[2]
		bbox = i[3]
		p = str(person.index(name))
		print p+'\t'+image+'\t'+bbox
		f.write(p+'\t'+image+'\t'+bbox+'\n')

	print 'Done in ..',time.time()-start