from utils import *

# mode=1
title = 'the exp result on 1/4 dataset'
name = ['fine grained docs interaction', 'docs interaction', 'w/o docs interaction']
x = [[1, 2, 3, 4, 5],
     [1, 2, 3, 4, 5],
     [1, 2, 3, 4, 5]]
y = [[0.6657827813909035, 0.6760045602032572, 0.6708645117602613, 0.6670647353197338, 0.6592922171781104],
    [0.6664454203246599, 0.6715599473914983, 0.6602006812468147, 0.6577879982426165, 0.6511086785907565],
    [0.6638356950291799, 0.6445899047335729, 0.6639940728449892, 0.655322273966451, 0.6432766495167762]]
x_scale = 1
y_scale = 0.01
label_x = 'epoch'
label_y = 'MAP'
path = 'result/compare.png'

draw_line_chart(title=title, note_list=name, x=x, y=y, x_scale=x_scale, y_scale=y_scale, label_x=label_x,
                label_y=label_y, path=path)















