import matplotlib.pyplot as plt

import matplotlib
import random
import matplotlib.pyplot as plt

# 中文乱码和坐标轴负号处理。
# matplotlib.rc('font', family='SimHei')
# plt.rcParams['axes.unicode_minus'] = False

# 城市数据。
model = ['RNN', 'LSTM', 'Seq2Seq', 'DAE-Seq2Seq', 'Attention-Seq2Seq','Our proposed method']
time = [1012,1231,3011,2297,3147,2328]
# 数组反转。
# model.reverse()

# 装载随机数据。
fig, ax = plt.subplots()
b = ax.barh(model, left=0, height=0.5, width=time)
# b = ax.barh(range(len(model)), time, color='#6699CC',width=0.5)

# 为横向水平的柱图右侧添加数据标签。
for rect in b:
    w = rect.get_width()
    ax.text(w, rect.get_y() + rect.get_height() / 2, '%d' %
            int(w), ha='left', va='center',fontsize=10)

# 设置Y轴纵坐标上的刻度线标签。
ax.set_yticks(range(len(model)))
ax.set_yticklabels(model)

plt.xticks((0,4000))
plt.xlabel('time(s)')
# plt.title('水平横向的柱状图', loc='center', fontsize='25',
#           fontweight='bold', color='red')

plt.show()
