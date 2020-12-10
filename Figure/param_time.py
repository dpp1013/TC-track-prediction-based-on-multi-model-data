import matplotlib.pyplot as plt
import numpy as np

a = ["ResNet-2","ResNet-3","ResNet-4","ResNet-5","ResNet-6"]
b = [0.0045,0.0042,0.0072,0.0088,0.019]
plt.figure()
plt.plot(a,b,'*-')
plt.xlabel('Model')
plt.ylabel('Loss')
# plt.figure(figsize=(14,8))
# x_t = range(len(a))
# plt.yticks(x_t,a)
# # plt.bar(a,b)
# plt.barh(a,b,height=0.35)
plt.show()