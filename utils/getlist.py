import os

path = 'autodl-tmp/Columbia_Gaze_Data_Set/'
folders = os.listdir(path)

text=open('list.txt','w')

for folder in folders:
    if (folder[0]=='.') :
        continue
    secondary_folder=os.path.join(path,folder)
    pics = os.listdir(secondary_folder)
    for pic in pics:
        if (pic[0]!='0') : continue
        text.write(os.path.join(secondary_folder,pic))
        text.write('\n')

text.close()