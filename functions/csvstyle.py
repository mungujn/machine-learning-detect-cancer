from functions import data
import numpy as np
import csv

x, y, x_test, y_test = data.getData()
x = np.reshape(x, (x.shape[0], -1))
dimensions = x.shape[1]

x.astype()

np.savetxt('lesions.csv',x_test, fmt='%.2f', delimiter=',')

# limit = 2
# lines = []
#
# for entry in range(limit):
#     line = ''
#     for dimension in range(dimensions):
#         line = line + str(x[entry][dimension]) + ','
#     type = y[entry]
#     if type == 0:
#         line = line + ',' + 'B'
#     elif type == 1:
#         line = line + ',' + 'M'
#     lines.append(line)
