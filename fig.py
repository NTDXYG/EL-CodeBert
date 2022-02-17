import matplotlib.pyplot as plt
from matplotlib import cm


def draw_bar(key_name, key_values):
    plt.rcParams['axes.unicode_minus'] = False
    # 标准柱状图的值
    def autolable(rects):
        for rect in rects:
            height = rect.get_height()
            if height >= 0:
                plt.text(rect.get_x() + rect.get_width() / 2.0 - 0.3, height + 0.02, '%.3f' % height)
            else:
                plt.text(rect.get_x() + rect.get_width() / 2.0 - 0.3, height - 0.06, '%.3f' % height)
                # 如果存在小于0的数值，则画0刻度横向直线
                plt.axhline(y=0, color='black')
    # 归一化
    norm = plt.Normalize(0, max(key_values))
    norm_values = norm(key_values)
    map_vir = cm.get_cmap(name='GnBu')
    colors = map_vir(norm_values)
    fig = plt.figure()  # 调用figure创建一个绘图对象
    plt.subplot(111)
    ax = plt.bar(key_name, key_values, width=0.5, color=colors, edgecolor='black')  # edgecolor边框颜色

    sm = cm.ScalarMappable(cmap=map_vir, norm=norm)  # norm设置最大最小值
    sm.set_array([])
    plt.colorbar(sm)
    plt.xticks(rotation=30)
    # autolable(ax)
    plt.show()


if __name__ == '__main__':
    key_name = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6', 'Layer 7', 'Layer 8', 'Layer 9', 'Layer 10', 'Layer 11', 'Layer 12', ]
    key_values = [0.1, 0.9, 1, 1, 0.4, 0.3, 0.1, 0.6, 0.2, 0.4, 0.5, 0.99]
    draw_bar(key_name, key_values)