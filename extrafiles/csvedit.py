import pandas as pd
import time
import re
import random
# reader1 = pd.read_csv('../traintest/dev_urls.txt', comment='#', sep='\t')
# reader2 = pd.read_csv('../traintest/eval_urls.txt', comment='#', sep='\t')
print 'extracting'
start = time.time()
reader3 = pd.read_csv('../traintest/classtrain.txt', sep=',')
print 'done in ..', time.time()-start

with open('../traintest/train.txt', 'w') as f:
	print 'file opened'
	f.write('person'+'\timage'+'\tbbox'+'\n')
	count = 0
	person = []
	names = []
	images = []
	bboxs = []
	print 'reading persons '
	start = time.time()
	for i in reader3.itertuples():
		if i[2] not in person:
			person += [i[2]]
		names += [[i[1], i[2]]]
		# print person+'\t'+image+'\t'+bbox
		# f.write(person+'\t'+image+'\t'+bbox+'\n')
	print 'people read in ..', time.time()-start

	print len(names)

	print 'writing to file'
	start = time.time()
	random.shuffle(names)
	for i in xrange(len(names)):
		image = names[i][0]
		name = names[i][1]
		# bbox = names[i][2]
		print image, name
		image = "../newtest/"+name+"/" + str(image)+'.jpg'
		p = str(person.index(name))
		if image!=None:
			image = re.sub('\.\./','',image)
			print name+'\t'+p+'\t'+image+'\t'
			f.write(name+'\t'+p+'\t'+image+'\t'+'\n')
		# r = randint(0,len(names))


	print 'Done in ..',time.time()-start