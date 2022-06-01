# 图像显示

from visdom import Visdom
import numpy as np

image = np.random.randn(6, 3, 200, 300) # 此时batch为6
viz = Visdom(env='my_image') # 注意此时我已经换了新环境
viz.images(image, win='x')

# 'x'
