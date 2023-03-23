from beat_tracking.data_loading import *
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def plot_pianorolls(args):
    
    dataset = args.dataset

    data = MyDataModule(args)

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    
    for el in train_loader:
        print(el[0])
        fig, ax = plt.subplots(1, figsize=(16, 4))
        ax.imshow(el[4][0].T, origin="lower", cmap='gray', interpolation='nearest', aspect='auto')
        plt.vlines(el[1]*100,0,88)
        plt.savefig("pianorolls/"+dataset+"_"+el[0][0].split("/")[-1][:-4]+".png")
        
        fig, ax = plt.subplots(1, figsize=(16, 4))
        ax.imshow(el[5][0].T, origin="lower", cmap='gray', interpolation='nearest', aspect='auto')
        plt.vlines(el[1]*100,0,88)
        plt.savefig("pianorolls/"+dataset+"_"+el[0][0].split("/")[-1][:-4]+"_pm.png")
        break
        
        #Save labels:
        #with open("amaps_labels/"+el[0][0].split("/")[-1][:-4]+".txt", 'w') as fp:
        #    for item in el[1][0]:
        #    # write each item on a new line
        #        fp.write("%s\n" % item.item())
    
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot pianorolls.')

    parser.add_argument('--dataset', type=str, help='Which dataset?')
    parser.add_argument('--mode', type=str, help='ismir/pm2s')

    args = parser.parse_args()

    plot_pianorolls(args)