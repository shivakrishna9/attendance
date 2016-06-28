# from recognise.resnet import *
from recognise.class_net import *
from recognise.get_input import *
from extrafiles.utility import *
import sys
sys.setrecursionlimit(100000)

i=[]
for image in glob.glob("extras/newtest/myclass/*/*.jpg"):
        person = image.split('/')[3]
        i.append(person)

x = Counter(i)
up = list(set(i))
people = sorted(up)

# people = ['abdul_karim', 'abdul_wajid', 'abhishek_bhatnagar', 'abhishek_joshi', 'aditya', 'ahsan',
#  'akshat', 'aly', 'aman', 'ameen', 'antriksh', 'anzal', 'ashar', 'asif', 'avishkar', 'bushra', 
#  'chaitanya', 'dhawal', 'farhan', 'farheen', 'ghalib', 'habib', 'harsh', 'irfan_ansari', 
#  'jeevan', 'manaff', 'manish', 'maria', 'mehrab', 'mohib', 'naeem', 'nikhil_mittal', 'nikhil_raman',
#  'prerit', 'raghib_ahsan', 'rahul', 'ravi', 'rehan', 'rezwan', 'rubab', 'sachin', 'sahil', 'saif',
#  'saifur', 'sajjad', 'sana', 'sapna', 'sarah_khan', 'sarah_masud', 'sarthak', 'shadab', 'shafiya', 
#  'shahbaz', 'shahjahan', 'sharan', 'shivam', 'shoaib', 'shoib', 'shruti', 'suhani', 'sultana', 
#  'sunny', 'sushmita', 'tushar', 'umar', 'zeya', 'zishan']


rec = VGG()
rec.VGGNet(people, len(people))
rec.train()
