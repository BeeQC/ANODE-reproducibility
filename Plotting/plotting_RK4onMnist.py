import json
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

mainPath = Path('/yourpathhere/')
folderPath = mainPath.joinpath('img_results_2019-12-12_23-17_mnist_experiments')
filePath = folderPath.joinpath('losses_and_nfes.json')

with open(filePath) as f:
    d = json.load(f)
    print(d)
dicAnode5 = d[0]
dicAnode50 = d[1]
dicAnode100 = d[2]
dicNode = d[3]

accuracyAnode5 = np.mean(np.array(dicAnode5['epoch_accuracy_history']), axis=0)
nfeAnode5 = np.mean(np.array(dicAnode5['epoch_nfe_history']), axis=0)
lossAnode5 = np.mean(np.array(dicAnode5['epoch_loss_history']), axis=0)

accuracyAnode50 = np.mean(np.array(dicAnode50['epoch_accuracy_history']), axis=0)
nfeAnode50 = np.mean(np.array(dicAnode50['epoch_nfe_history']), axis=0)
lossAnode50 = np.mean(np.array(dicAnode50['epoch_loss_history']), axis=0)

accuracyAnode100 = np.mean(np.array(dicAnode100['epoch_accuracy_history']), axis=0)
nfeAnode100 = np.mean(np.array(dicAnode100['epoch_nfe_history']), axis=0)
lossAnode100 = np.mean(np.array(dicAnode100['epoch_loss_history']), axis=0)

accuracyNode = np.mean(np.array(dicNode['epoch_accuracy_history']), axis=0)
nfeNode = np.mean(np.array(dicNode['epoch_nfe_history']), axis=0)
lossNode = np.mean(np.array(dicNode['epoch_loss_history']), axis=0)


epochs = np.arange(1, len(np.squeeze(accuracyAnode5))+1)
# recreating figures from the json files saved by the experimental runs - 5 augmented Dimensions
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3.5))
ax1.plot(epochs, np.squeeze(accuracyNode), label='p=0')
ax1.plot(epochs, np.squeeze(accuracyAnode5),    label='p=5')
ax1.plot(epochs, np.squeeze(accuracyAnode50),   label='p=50')
ax1.plot(epochs, np.squeeze(accuracyAnode100),  label='p=100')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
# ax1.set_ylim([0.75, 1.05])
ax1.legend()

ax2.scatter(nfeNode, accuracyNode, label='p=0')
ax2.scatter(nfeAnode5, accuracyAnode5,  label='p=5')
ax2.scatter(nfeAnode50, accuracyAnode50,  label='p=50')
ax2.scatter(nfeAnode100, accuracyAnode100,  label='p=100')
ax2.set_xlabel('NFE')
# ax2.set_xlim([20, 130])
# ax2.set_ylim([0.75, 1.05])
ax2.legend()

ax3.plot(epochs, np.squeeze(lossNode), label='p=0')
ax3.plot(epochs, np.squeeze(lossAnode5),    label='p=5')
ax3.plot(epochs, np.squeeze(lossAnode50),   label='p=50')
ax3.plot(epochs, np.squeeze(lossAnode100),  label='p=100')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('loss')
# plt.xlim([0, 130])
ax3.legend()
plt.tight_layout()
plt.savefig('RK4_Dim_MNIST.png', format='png', dpi=400, bbox_inches='tight')

