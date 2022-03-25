import argparse
import pickle

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='ntu/xview', choices={'kinetics', 'ntu/xsub', 'ntu/xview','ntu120/xsub'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
arg = parser.parse_args()

dataset = arg.datasets
label = open('./data/' + dataset + '/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open('./work_dir/ntu_softmax/xview'  + '/sgcn_test_joint_fourpart_softamx_43_128/epoch1_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('./work_dir/ntu_softmax/xview'  + '/sgcn_test_bone_fourpart_softmax_48_128/epoch1_test_score.pkl', 'rb')
r2 = list(pickle.load(r2).items())
r3 = open('./work_dir/ntu_softmax/xview' +  '/sgcn_test_joint_motion_fourpart_softmax_32/epoch1_test_score.pkl', 'rb')
r3 = list(pickle.load(r3).items())
r4 = open('./work_dir/ntu_softmax/xview'  + '/sgcn_test_bone_motion_fourpart_softmax_128/epoch1_test_score.pkl', 'rb')
r4 = list(pickle.load(r4).items())
r5 = open('./work_dir/ntu_softmax/xview'  + '/sgcn_test_starin_fourpart_softmax_18/epoch1_test_score.pkl', 'rb')
r5 = list(pickle.load(r5).items())
r6 = open('./work_dir/ntu_softmax/xview'  + '/sgcn_test_starout_fourpart_softmax_128/epoch1_test_score.pkl', 'rb')
r6 = list(pickle.load(r6).items())
right_num = total_num = right_num_5 = 0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    _, r33 = r3[i]
    _, r44 = r4[i]
    _, r55 = r5[i]
    _, r66 = r6[i]
    r = r11 + r22 +r55+r66* arg.alpha
    #r = r55 + r66 +r33 + r44 * arg.alpha
    #r = r11 + r22 +r33 + r44  + r55 + r66* arg.alpha
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num
print(acc, acc5)
