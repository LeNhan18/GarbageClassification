import matplotlib.pyplot as plt
import numpy as np
cate = np.random.rand(50)
values = 2*cate+1+0.1* np.random.rand(50)
plt.scatter(cate , values)
plt.title('Nhanle')
plt.xlabel('cate')
plt.ylabel('values')
plt.show()