import pandas as pd

filereader = pd.read_csv('lfwpeople.txt',iterator='True',chunksize=10,names=['name','class'])

columns = ['name','class']

for row in filereader:
    row[:2].to_csv('1.csv',header=False,index=0,mode='a')
    row[2:4].to_csv('2.csv',header=False,index=0,mode='a')
    row[4:6].to_csv('3.csv',header=False,index=0,mode='a')
    row[6:8].to_csv('4.csv',header=False,index=0,mode='a')
    row[8:].to_csv('5.csv',header=False,index=0,mode='a')
#     break

# for row in filereader:
#     row[:2].to_csv('1.csv',header=False,index=0,mode='a')
#     row[2:4].to_csv('2.csv',header=False,index=0,mode='a')
#     row[4:6].to_csv('3.csv',header=False,index=0,mode='a')
#     row[6:8].to_csv('4.csv',header=False,index=0,mode='a')
#     row[8:].to_csv('5.csv',header=False,index=0,mode='a')
    