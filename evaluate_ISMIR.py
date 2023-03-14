import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import madmom
import partitura as pt
import torch
import warnings
from beat_tracking.helper_functions import *
from beat_tracking.data_loading import *
from beat_tracking.models import MyMadmom

import pytorch_lightning as pl
# TESTING

checkpoint_list = ["models_ISMIR/model_iter_96000.pt"]

def evaluate(args):
    data = MyDataModule(args)
    
    results_dir = args.results_dir
    dataset = args.dataset
    
    val_loader = data.val_dataloader()

    for checkpoint in checkpoint_list:
        try:
            model = torch.load(checkpoint,map_location = torch.device("cpu"))
        except Exception as e:
            print(checkpoint,"error:\n",e)
            continue
        model.eval()
        warnings.filterwarnings("ignore")
        
        device = next(model.parameters()).device
        print("\nCheckpoint:",checkpoint,"--- Device:",device,"\n")

        scores = []
        with torch.no_grad():
            for j, datapoint_test in enumerate(val_loader):

                padded_array_test = cnn_pad(datapoint_test[4].float(),2)
                padded_array_test = torch.tensor(padded_array_test)
                
                print("Test sample number",j+1,"/",len(val_loader),"---",datapoint_test[0][0],"--- Input shape:",datapoint_test[4].shape)
                outputs = model(padded_array_test)

                #Calculate F-Score:
                try:
                    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
                    beat_times = proc(outputs[0][:,0].detach().cpu().numpy())
                    evaluate = madmom.evaluation.beats.BeatEvaluation(beat_times, datapoint_test[1][0])
                    f_score_test = evaluate.fmeasure
                except Exception as e:
                    print("Sample can not be processed correctly. Error in beat process:",e)
                    f_score_test = 0

                scores.append(j,datapoint_test[0][0],f_score_test)
                if j % 10 == 0:
                    print(j,datapoint_test[0][0],f_score_test)
        
        with open(results_dir+"/evaluate_ISMIR_"+dataset+"_"+"val"+".txt","w") as fp:
            sum = 0
            for entry in scores:
                fp.write("%s\t%s\t%s\n" % (entry[0],entry[1],entry[2]))
                sum += entry[2]
            fp.write("%s\n" % "Summary: "+str(sum/len(scores)))
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate ISMIR.')

    parser.add_argument('--results_dir', type=str, help='Where to store the results.')
    parser.add_argument('--dataset', type=str, help='Which dataset?')
    parser.add_argument('--mode', type=str, help='ismir/pm2s')

    args = parser.parse_args()

    evaluate(args)