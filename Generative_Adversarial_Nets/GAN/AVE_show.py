import matplotlib.pyplot as plt
from matplotlib import image as mpimg

reconsPath = './ave_samples/reconst-30.png'
Image = mpimg.imread(reconsPath)
plt.imshow(Image)  # 显示图片
plt.axis('off')  # 不显示坐标轴
plt.show()

genPath = './ave_samples/sampled-30.png'
Image = mpimg.imread(genPath)
plt.imshow(Image)  # 显示图片
plt.axis('off')  # 不显示坐标轴
plt.show()