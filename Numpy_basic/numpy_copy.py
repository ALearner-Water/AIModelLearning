#numpy复制
import numpy as np
a=np.arange(0,4)
print(a)
#此时给b c d赋值之后再修改a，bc也会跟着变
b=a
c=b
print(a,b,c)
a[0]=5
b[1]=9
print(a,b,c)
print(a is b is c)  #三个是一模一样的

#使用a.copy()或者np.copy(a)来让a只赋值不关联
d=np.copy(a)    #deep copy
e=a.copy()
print(a,d,e)
#此时修改a的值，其他不会关联
a[0]=6
print(a,d,e)