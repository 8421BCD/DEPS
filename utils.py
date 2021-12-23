import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def draw_line_chart(title, note_list, x, y, x_scale, y_scale, label_x, label_y, path = None):
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus']=False
    plt.switch_backend('agg')
    for i in range(len(x)):
        plt.plot(x[i], y[i], marker='', mec='r', mfc='w', label=note_list[i], linewidth=2)
    plt.legend(fontsize=16)  # 让图例生效
    # plt.xticks(x, note_list, rotation=45)
    plt.margins(0)
    plt.xlabel(label_x, fontsize=15)  # X轴标签
    plt.ylabel(label_y, fontsize=16)  # Y轴标签
    plt.title(title, fontsize=14)  # 标题
    plt.tick_params(labelsize=14)

    # ax.set_xlabel(label_x, fontsize=15)
    # ax.set_ylabel(label_y, fontsize=16)
    # ax.tick_params(axis='x', labelsize=14)
    # ax.tick_params(axis='y', labelsize=14)
    # ax.legend(fontsize=14)  # 让图例生效



    # 设置x轴的刻度间隔，并存在变量里
    x_major_locator = MultipleLocator(x_scale)
    # 把y轴的刻度间隔设置为10，并存在变量里
    y_major_locator = MultipleLocator(y_scale)
    # ax为两条坐标轴的实例
    ax = plt.gca()
    # 把x轴的主刻度设置为1的倍数
    ax.xaxis.set_major_locator(x_major_locator)
    # 把y轴的主刻度设置为10的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    #范围
    plt.xlim(min(x[0]), max(x[-1]))
    plt.ylim(0.64, 0.68)

    if path:
        plt.savefig(path)
    plt.show()
