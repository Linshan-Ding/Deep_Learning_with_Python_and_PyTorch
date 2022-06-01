from matplotlib import image as mpimg, pyplot as plt

reconsPath = './gan_samples/real_images.png'
Image = mpimg.imread(reconsPath)
plt.imshow(Image)  # 显示图片
plt.axis('off')  # 不显示坐标轴
plt.show()

reconsPath = './gan_samples/fake_images-100.png'
Image = mpimg.imread(reconsPath)
plt.imshow(Image)  # 显示图片
plt.axis('off')  # 不显示坐标轴
plt.show()