
import numpy as np
import matplotlib.pyplot as plt
students = ['學生', '老師', '天氣', '季節']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
math_scores = [78, 67, 90, 81]
x = np.arange(len(students))
plt.bar(x, math_scores, color=['red', 'green', 'blue', 'yellow'])
plt.xticks(x, students)
plt.xlabel('Students')
plt.ylabel('Math')
plt.title('Final Term')
plt.show()

from matplotlib.font_manager import FontProperties
import matplotlib.font_manager
from matplotlib.font_manager import FontProperties

# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.plot((1,2,3),(4,3,-1))
# plt.title("聲量圖")
# plt.ylabel("文章數量")
# plt.xlabel("品牌名稱")
# plt.show()