import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
a=np.array([23,45,16,77])
b=np.array([2,3,1,4])
data=pd.DataFrame({'s':b,'ro':a})
data.plot()
#plt.show()
fir=plt.figure()
plt.bar(data['s'],data['ro'])
plt.xlabel('speed')
plt.ylabel('rotation')
#plt.show()
plt.subplot()
data.hist()
plt.show()
print(data[data['s']<3])