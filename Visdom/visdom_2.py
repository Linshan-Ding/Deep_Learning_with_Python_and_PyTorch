# 多条曲线实时曲线绘制

from visdom import Visdom

'''
多条曲线绘制 实际上就是传入y值时为一个向量
'''
viz = Visdom(env='my_wind') # 注意此时我已经换了新环境
#设置起始点
viz.line([[0.0,0.0]],    ## Y的起始点
        [0.],    ## X的起始点
        win="test loss",    ##窗口名称
        opts=dict(title='test_loss')  ## 图像标例
        )
'''
模型数据
'''
viz.line([[1.1,1.5]],   ## Y的下一个点
        [1.],   ## X的下一个点
        win="test loss", ## 窗口名称
        update='append'   ## 添加到上一个点后面
        )

# 'test loss'
