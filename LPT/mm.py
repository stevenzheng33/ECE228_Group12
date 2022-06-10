# import torch
# torch.cuda.empty_cache()
# import gc

# gc.collect()
# torch.cuda.memory_summary(device=None, abbreviated=False)
import matplotlib.pyplot as plt
import numpy
import pandas as pd
res = pd.read_csv('log.txt',delim_whitespace=True, skiprows=5,header = None)
for i in res[4]:
    i = i*2.4
readfile = open(r"C:\Users\kicph\Desktop\228\report_code\log(1).txt") 
fline = readfile.readlines()
fline = fline[5::]
print(fline)
# epo = list(range(res[5].size))
# fig = plt.figure()
# plt.subplot(1, 2, 1)
# plt.plot(epo,res[5],label = 'valid_acc')
# plt.plot(epo,res[6],label = 'test_acc')
# plt.xlabel('epoch')
# plt.ylabel('accuracy line chart')
# plt.subplot(1, 2, 2)
# plt.plot(epo,res[4])
# plt.xlabel('epoch')
# plt.ylabel('loss function')