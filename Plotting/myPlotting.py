import json
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

mainPath = Path('/yourpathhere/')
folderPath5 = mainPath.joinpath('MNIST_5Dim')
folderPath50 = mainPath.joinpath('MNIST_50Dim')
folderPath75 = mainPath.joinpath('MNIST_75Dim')
folderPath100 = mainPath.joinpath('MNIST_100Dim')

filePath5 = folderPath5.joinpath('losses_and_nfes.json')
filePath50 = folderPath50.joinpath('losses_and_nfes.json')
filePath75 = folderPath75.joinpath('losses_and_nfes.json')
filePath100 = folderPath100.joinpath('losses_and_nfes.json')

with open(filePath5) as f:
    d = json.load(f)
    print(d)
dicAnode = d[0]
dicNode = d[1]
accuracyAnode5 = np.array(dicAnode['epoch_accuracy_history'])
nfeAnode5 = np.array(dicAnode['epoch_total_nfe_history'])
lossAnode5 = np.array(dicAnode['epoch_loss_history'])

with open(filePath50) as f:
    d = json.load(f)
    print(d)
dicAnode = d[0]
accuracyAnode50 = np.array(dicAnode['epoch_accuracy_history'])
nfeAnode50 = np.array(dicAnode['epoch_total_nfe_history'])
lossAnode50 = np.array(dicAnode['epoch_loss_history'])

with open(filePath75) as f:
    d = json.load(f)
    print(d)
dicAnode = d[0]
accuracyAnode75 = np.array(dicAnode['epoch_accuracy_history'])
nfeAnode75 = np.array(dicAnode['epoch_total_nfe_history'])
lossAnode75 = np.array(dicAnode['epoch_loss_history'])

with open(filePath100) as f:
    d = json.load(f)
    print(d)
dicAnode = d[0]
accuracyAnode100 = np.array(dicAnode['epoch_accuracy_history'])
nfeAnode100 = np.array(dicAnode['epoch_total_nfe_history'])
lossAnode100 = np.array(dicAnode['epoch_loss_history'])

accuracyNode = np.array(dicNode['epoch_accuracy_history'])
nfeNode = np.array(dicNode['epoch_total_nfe_history'])
lossNode = np.array(dicNode['epoch_loss_history'])

epochs = np.arange(1, len(np.squeeze(accuracyAnode5))+1)
# recreating figures from the json files saved by the experimental runs - 5 augmented Dimensions
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))
ax1.plot(epochs, np.squeeze(accuracyAnode5), label='ANODE')
ax1.plot(epochs, np.squeeze(accuracyNode), label='NODE')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.set_ylim([0.75, 1.05])
ax1.legend()

ax2.scatter(nfeAnode5, accuracyAnode5, label='ANODE')
ax2.scatter(nfeNode, accuracyNode, label='NODE')
ax2.set_xlabel('NFE')
ax2.set_xlim([20, 130])
ax2.set_ylim([0.75, 1.05])
ax2.legend()

ax3.plot(epochs, np.squeeze(lossAnode5), label='ANODE')
ax3.plot(epochs, np.squeeze(lossNode), label='NODE')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('loss')
# plt.xlim([0, 130])
ax3.legend()
plt.tight_layout()
plt.savefig('test5Dim.png', format='png', dpi=400, bbox_inches='tight')

# recreating figures from the json files saved by the experimental runs - 5 augmented Dimensions
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))
ax1.plot(epochs, np.squeeze(accuracyAnode5),    label='p=5')
ax1.plot(epochs, np.squeeze(accuracyAnode50),   label='p=50')
ax1.plot(epochs, np.squeeze(accuracyAnode75),   label='p=75')
ax1.plot(epochs, np.squeeze(accuracyAnode100),  label='p=100')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.set_ylim([0.75, 1.05])
ax1.legend()

ax2.scatter(nfeAnode5,   accuracyAnode5,   label='p=5')
ax2.scatter(nfeAnode50,  accuracyAnode50,  label='p=50')
ax2.scatter(nfeAnode75,  accuracyAnode75,  label='p=75')
ax2.scatter(nfeAnode100, accuracyAnode100, label='p=100')
ax2.set_xlabel('NFE')
ax2.set_xlim([20, 130])
ax2.set_ylim([0.75, 1.05])
ax2.legend()

ax3.plot(epochs, np.squeeze(lossAnode5), label='p=5')
ax3.plot(epochs, np.squeeze(lossAnode50), label='p=50')
ax3.plot(epochs, np.squeeze(lossAnode75), label='p=75')
ax3.plot(epochs, np.squeeze(lossAnode100), label='p=100')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('loss')
# plt.xlim([0, 130])
ax3.legend()
plt.tight_layout()
plt.savefig('augmentedDimensionComparison.png', format='png', dpi=400, bbox_inches='tight')
