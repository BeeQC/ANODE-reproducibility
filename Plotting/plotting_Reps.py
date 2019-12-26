import json
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

mainPath = Path('/yourpathhere/')
folderPath100 = mainPath.joinpath('100reps/0/')
filePath100 = folderPath100.joinpath('model_losses.json')


with open(filePath100) as f:
    d = json.load(f)
    print(d)
dicAnode = d[2]
dicNode = d[1]
accuracyAnode100 = np.array(dicAnode['epoch_accuracy_history'])
nfeAnode100 = np.array(dicAnode['epoch_nfe_history'])
lossAnode100 = np.array(dicAnode['epoch_loss_history'])

accuracyNode100 = np.array(dicNode['epoch_accuracy_history'])
nfeNode100 = np.array(dicNode['epoch_nfe_history'])
lossNode100 = np.array(dicNode['epoch_loss_history'])

epochs = np.arange(1, 30+1)
# recreating figures from the json files saved by the experimental runs - 5 augmented Dimensions
fig1, ([ax1, ax2]) = plt.subplots(1, 2, figsize=(9, 3))

shadeNode = np.std(accuracyNode100, axis=0)
shadeAnode = np.std(accuracyAnode100, axis=0)
ax1.plot(epochs, np.squeeze(np.mean(accuracyNode100, axis=0)), label='NODE')
ax1.fill_between(epochs,  np.squeeze(np.mean(accuracyNode100, axis=0)) - shadeNode,  np.squeeze(np.mean(accuracyNode100, axis=0)) + shadeNode, alpha=0.5)
ax1.plot(epochs, np.squeeze(np.mean(accuracyAnode100, axis=0)), label='ANODE')
ax1.fill_between(epochs,  np.squeeze(np.mean(accuracyAnode100, axis=0)) - shadeAnode,  np.squeeze(np.mean(accuracyAnode100, axis=0)) + shadeAnode, alpha=0.5)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()

shadeNode = np.std(lossNode100, axis=0)
shadeAnode = np.std(lossAnode100, axis=0)
ax2.plot(epochs, np.squeeze(np.mean(lossNode100, axis=0)), label='NODE')
ax2.fill_between(epochs,  np.squeeze(np.mean(lossNode100, axis=0)) - shadeNode,  np.squeeze(np.mean(lossNode100, axis=0)) + shadeNode, alpha=0.5)
ax2.plot(epochs, np.squeeze(np.mean(lossAnode100, axis=0)), label='ANODE')
ax2.fill_between(epochs,  np.squeeze(np.mean(lossAnode100, axis=0)) - shadeAnode,  np.squeeze(np.mean(lossAnode100, axis=0)) + shadeAnode, alpha=0.5)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.savefig('100reps_SphericalDataSet.png', format='png', dpi=400, bbox_inches='tight')
