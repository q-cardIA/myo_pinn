# import necessary modules
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt 
import os, csv
from scipy.stats import mannwhitneyu

from src.utils.data_functions.make_images import show_maps, training_visualization

# Figure 3 and appendix C
for j in range(1,9):
    show_maps(str(j),('pt'+str(j) + '_1.tf',
            'pt'+str(j) + '_2.tf',
            'pt'+str(j) + '_3.tf'), 0)

# Figure 4 
training_visualization(["myo_pinn_20201112-044948/events.out.tfevents.1605156588.gpuserver01.27697.485950179.v2"]) 
plt.show()


# Figure 5
visual = []
with open(os.path.join("files","visual_assessment.csv")) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        visual.append(row[1:])

flows = np.load(os.path.join("files","flows.npy")) * 1.05 # the scaling is the specific density of the myocardium 1.05 g/ml
vps = np.load(os.path.join("files","vps.npy"))
ves = np.load(os.path.join("files","ves.npy"))
perms = np.load(os.path.join("files","perms.npy")) * 1.05

pos = []
neg = []
for i in range(8):
    this_flow = flows[i]
    this_visual = visual[i]
    for j in range(16):
        if this_visual[j] == '1':
            pos.append(this_flow[j]) 
        else:
            neg.append(this_flow[j])



mpl.style.use('seaborn-darkgrid')    
box1 = plt.boxplot([pos,neg],patch_artist=True,showfliers=False)
plt.plot(np.random.normal(1, 0.06, size=len(pos)), pos, 'o', color='grey')
plt.plot(np.random.normal(2, 0.06, size=len(neg)), neg, 'o', color='grey')
plt.setp(box1["boxes"], facecolor='orange')
plt.setp(box1["boxes"], alpha=0.4)
plt.xticks([1,2], ('Ischaemia','No ischaemia'), fontsize=16,fontweight='bold',fontname='Calibri')
plt.yticks(fontsize=12)
plt.ylabel('MBF (ml/min/g)',fontsize=16,fontweight='bold',fontname='Calibri')
plt.show()


# Numerical results
print(np.median(pos))
print(np.percentile(pos,25))
print(np.percentile(pos,75))

print(np.median(neg))
print(np.percentile(neg,25))
print(np.percentile(neg,75))

print(mannwhitneyu(pos, neg, use_continuity=True,alternative='less'))



# Table 2
print(np.median(np.asarray(flows)))
print(np.percentile(np.asarray(flows),25))
print(np.percentile(np.asarray(flows),75))

print(np.median(np.asarray(vps)))
print(np.percentile(np.asarray(vps),25))
print(np.percentile(np.asarray(vps),75))

print(np.median(np.asarray(ves)))
print(np.percentile(np.asarray(ves),25))
print(np.percentile(np.asarray(ves),75))

print(np.median(np.asarray(perms)))
print(np.percentile(np.asarray(perms),25))
print(np.percentile(np.asarray(perms),75))
