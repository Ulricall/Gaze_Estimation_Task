'''
To generate the label of the Columbia Gaze Data Set.
'''

import os
import math

if __name__ == '__main__':
    path = '../input/gazeestimate/Columbia_Gaze_Data_Set/'
    folders = os.listdir(path)

    text = open('../input/gazeestimate/Columbia_Gaze_Data_Set/list.txt', 'w')

    for folder in folders:
        if(folder[0] == '.'):
            continue
        secondary_folder = os.path.join(path, folder)
        pics = os.listdir(secondary_folder)
        for pic in pics:
            if (pic[0] != '0'): continue
            
            pic_split = pic.split("_")
            V_label = str(float(pic_split[3].strip('V')) / 360 * (2*math.pi))
            H_label = str(float(pic_split[4].strip('H.jpg')) / 360 * (2*math.pi))
            label = [os.path.join(folder, pic), H_label, V_label] 

            text.write(" ".join(label))
            text.write('\n')

    text.close()