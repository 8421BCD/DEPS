from utils import *

# mode=1
title = 'the exp result on 1/4 dataset'
name = ['fine-grained docs interaction', 'coarse-grained docs interaction', 'w/o docs interaction']
x = [[1, 2, 3, 4, 5],
     [1, 2, 3, 4, 5],
     [1, 2, 3, 4, 5]]
y = [[0.6689830202888779, 0.6744318803144002, 0.6719721552974345, 0.6668848693805249, 0.6575581911736341],
    [0.6566672666925178, 0.6704511925414015, 0.6629904018389428, 0.6467965232293498, 0.6439764135136855],
    [0.6567964867758934, 0.6611170461675375, 0.6569555846873736, 0.6452658122075344, 0.6421829807494648]]
x_scale = 1
y_scale = 0.01
label_x = 'epoch'
label_y = 'MAP'
path = 'result/compare.png'

draw_line_chart(title=title, note_list=name, x=x, y=y, x_scale=x_scale, y_scale=y_scale, label_x=label_x,
                label_y=label_y, path=path)















