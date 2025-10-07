---
date:
  created: 2025-10-06
title: python画图
tags:
    - python
---

python基本画图实现
<!-- more -->

## python画图

注意引入相应的库
```py
import numpy as np
import matplotlib.pyplot as plt
```

### 单张图
```py
# 在图中从位置(0,0)到位置(6,250)画一条线  
xpoints = np.array([0, 6])  
ypoints = np.array([0, 250])  
plt.plot(xpoints, ypoints)  
plt.show()

# 不指定x轴的点，默认为0到1平均分
ypoints = np.array([0, 250])
plt.plot(ypoints)  
plt.show()
```

### 多点图
```py
xpoints = np.array([33, 7, 6, 13])  
ypoints = np.array([3, 23, 88, 42])  
plt.plot(xpoints, ypoints)  
plt.show()
```

### 强调连接点
```py
xpoints = np.array([1, 3, 5, 7])  
ypoints = np.array([3, 23, 88, 42])  
# plt.plot(xpoints, ypoints, marker='o')  
plt.plot(xpoints, ypoints, marker='*')  
plt.show()
```

### 颜色参考

|字符|颜色|
|:-:|:-:|
|b|蓝色|
|g|绿色|
|r|红色|
|c|青色|
|m|品红色|
|y|黄色|
|k|黑色|
|w|白色|

### 格式化字符串设置曲线
按照marker|line|color的顺序

```py
xpoints = np.array([1, 3, 5, 7])  
ypoints = np.array([3, 23, 88, 42])  
plt.plot(xpoints, ypoints, 'o:r')  
plt.show()
```

### 设置标记尺寸
ms或者marksize

```py
xpoints = np.array([1, 3, 5, 7])  
ypoints = np.array([3, 23, 88, 42])  
plt.plot(xpoints, ypoints, 'o:r',ms='20')  
plt.show()
```

### 同时画两条曲线
```py
xpoints = np.array([1, 3, 5, 7])  
ypoints1 = np.array([3, 23, 88, 42])  
ypoints2 = np.array([78, 13, 44, 99])  
plt.plot(xpoints, ypoints1)  
plt.plot(xpoints, ypoints2)  
plt.show()
```

### 设置标签与标题
注意要设置字体

```py
# 设置字体为楷体
matplotlib.rcParams['font.sans-serif'] = ['KaiTi']  

xpoints = np.array([1, 3, 5, 7])  
ypoints = np.array([78, 13, 44, 99])  
plt.plot(xpoints, ypoints)  
plt.xlabel('时间节点')  
plt.ylabel('收入')  
plt.title('标题')
plt.show()
```

### 添加网格线

`#!python plt.grid()`













