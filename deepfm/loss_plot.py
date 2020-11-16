import numpy as np
import matplotlib.pyplot as plt
import json

class plot_curve(object):
    def __init__(self, json_data, save_name, save_file = True, dev_info = True):
        self.json_data = json_data
        #self.json_file = json_file
        self.save_name = save_name
        self.save_file = save_file
        self.dev_info = dev_info
        #with open(json_file, "r") as json_r:
        #    self.json_data = json.load(json_r)

    def plot_loss(self):
        train_loss = self.json_data["train_epoch_loss"]
        train_auc = self.json_data["train_epoch_auc"]
        x_loss =  list(range(len(train_loss)))
        x_auc = list(range(len(train_auc)))

        if self.dev_info:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize=(10, 7))
        else:
            fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 7))

        fig.suptitle("training curve", fontsize= 13, x = 0.53, y= 0.93)

        ax1.plot(x_loss, train_loss, color = "red")
        ax1.grid(linestyle='dotted')
        ax1.set_ylabel('training loss', fontsize = 12)
        #plt.tight_layout()

        ax2.plot(x_auc, train_auc, color = "green")
        ax2.grid(linestyle='dotted')
        ax2.set_ylabel('training auc', fontsize = 12)

        if self.dev_info:
            pass
        else:
            ax2.set_xlabel('epoch', fontsize = 12)

        #ax2.set_ylabel('training auc', fontsize = 12)
       # plt.tight_layout()

        if self.dev_info:
            dev_loss = self.json_data["dev_loss"]
            dev_auc = self.json_data["dev_auc"]
            x_dev_loss =  list(range(len(dev_loss)))
            x_dev_auc = list(range(len(dev_auc)))
            ax3.plot(x_dev_loss, dev_loss, color = "orange")
            ax3.grid(linestyle='dotted')
            ax3.set_ylabel('validation loss', fontsize = 12)
            #ax2.set_xlabel('epoch', fontsize = 12)

            ax4.plot(x_dev_auc, dev_auc, color = "blue")
            ax4.grid(linestyle='dotted')
            ax4.set_ylabel('validation auc', fontsize = 12)
            ax4.set_xlabel('epoch', fontsize = 12)

        if self.save_file == True:
            plt.savefig(self.save_name, dpi=600, bbox_inches='tight')
        #plt.show()
