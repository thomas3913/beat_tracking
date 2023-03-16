import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import sys
import time
import mir_eval
import madmom
import partitura as pt
import torch
import torch.nn as nn
import warnings
from beat_tracking.helper_functions import *
from beat_tracking.data_loading import *
from beat_tracking.models import MyMadmom
from beat_tracking.modules import MyMadmomModule

import pytorch_lightning as pl

warnings.filterwarnings("ignore")

# TRAINING LOOP

def train(args):
    data = MyDataModule(args)
    
    checkpoints_dir = args.checkpoints_dir
    figures_dir = args.figures_dir
    dataset = args.dataset
    validate_every = args.validate_every
    
    model = MyMadmomModule()
    
    trainer = pl.Trainer(
        
        #logger=logger,
        log_every_n_steps=50,
        reload_dataloaders_every_n_epochs=True,
        #gpus=gpus,
        accelerator='gpu',
        devices=[0]
    )
    
    trainer.fit(model, data)
    
    """
    
    train_loader = data.train_dataloader()
    validation_loader = data.val_dataloader()
    
    num_epochs = 6

    # Of the following 6 lines the first 4 are for loading a model, the last 2 for creating a new one.
    model_to_load = "models_ISMIR/checkpoints/asap/model_iter_8000.pt"
    model = torch.load(model_to_load)
    model_progress_info = model_to_load[:-3]+"_progress_info.json"
    loaded_model = True
    #model = MyMadmom()
    #loaded_model = False

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    # Load the progress statistics from file or create corresponding variables and lists if new model:
    if loaded_model == True:
        with open(model_progress_info, 'r') as f:
            load_dict = json.load(f)
        loaded_epoch = load_dict["epoch"]
        iter = load_dict["iter"]
        loss_list = load_dict["loss_list"]
        val_loss_list = load_dict["val_loss_list"]
        fscore_list_val_average = load_dict["fscore_list_val_average"]
        fscore_list_val_average_db = load_dict["fscore_list_val_average_db"]
        loaded_time = load_dict["total_time"]
        learning_rate = load_dict["learning_rate"]
        learning_rates_list = load_dict["learning_rates_list"]
    else:
        loaded_epoch, iter, loaded_time = 0, 0, 0
        loss_list, val_loss_list, fscore_list_val_average, fscore_list_val_average_db = list(), list(), list(), list()
        learning_rate = 0.001
        learning_rates_list = list()
        
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        model.cuda()
        with torch.no_grad():
            torch.cuda.empty_cache()
        
    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    samples_passed = 0
                                                                                
    begin_time = time.time()
    print("Training Started! Device:",device,"\nTrainable parameters:",params)

    for epoch in range(loaded_epoch, num_epochs+loaded_epoch):
        print("\nEPOCH " +str(epoch+1)+" of "+str(num_epochs+loaded_epoch)+"\n")

        loss_sum = 0
        sample_loss_values = list()
        
        for i,datapoint in enumerate(train_loader):

            model.train()
            sample_number = i+1
            
            if iter % 25 == 0:
                print("Sample",sample_number,"/",len(train_loader),"( Ep.",epoch+1,"- Iter.",iter+1,") ---",datapoint[0][0],"--- Input shape:",datapoint[4].shape)

            # Pad the input array so that the output and the labels have same shape:
            padded_array = cnn_pad(datapoint[4].float(),2)
            padded_array = torch.tensor(padded_array)

            if device == "cuda":
                datapoint[4] = datapoint[4].cuda()
                padded_array = padded_array.cuda()
            optimizer.zero_grad()
            
            outputs = model(padded_array)

            # Get beat activation function from the time annotations and widen beat targets for better accuracy:                
            beat_activation = madmom.utils.quantize_events(datapoint[1][0], fps=100, length=len(datapoint[4][0]))
            widen_beat_targets(beat_activation)
            beat_activation = torch.tensor(beat_activation)

            # Same for downbeats:
            beat_activation_db = madmom.utils.quantize_events(datapoint[2][0], fps=100, length=len(datapoint[4][0]))
            widen_beat_targets(beat_activation_db)
            beat_activation_db = torch.tensor(beat_activation_db)
            
            # Calculate two separate losses for beats and downbeats:
            loss_b = criterion(outputs[0][:,0].float(),beat_activation.float().to(device))
            loss_db = criterion(outputs[1][:,0].float(),beat_activation_db.float().to(device))
            
            # Calculate passed time and remaining time:
            timestamp = time.time()-begin_time
            total_time = timestamp + loaded_time
            d,h,m,s = show_time(total_time)
            samples_passed += 1
            sample_avg_time = timestamp / samples_passed
            remaining_time = ((len(train_loader) - sample_number) * sample_avg_time) + ((num_epochs+loaded_epoch - epoch - 1) * len(train_loader) * sample_avg_time)
            
            if iter % 25 == 0:
                print("Loss (b):","%.4f"%loss_b.detach(),"--- Loss (db):","%.4f"%loss_db.detach(),"--- Time:",str(time.strftime("%H:%M:%S", time.gmtime(timestamp))),"--- Total time:",d,"days,",f"{h:02d}:{m:02d}:{s:02d}","--- Remaining:",str(time.strftime("%H:%M:%S", time.gmtime(remaining_time))),"\n")

            # Add the two loss functions:
            loss = loss_b + loss_db
            loss_sum += loss.detach()

            loss.backward()
            optimizer.step()
            
            iter=iter+1

            model.eval()      

            with torch.no_grad():
                torch.cuda.empty_cache()
            
            # Evaluation on the validation set:
            if iter % int(validate_every) == 0:

                loss_list.append(loss_sum.cpu().item())
                
                # Plot training loss:
                plot_value_list(loss_list,"blue","Training loss after "+str(iter)+" samples",save=True,save_name=figures_dir+"Training_loss_"+dataset)

                loss_sum = 0
                val_loss = 0
                total = 0
                
                fscore_list_val = list()
                fscore_list_db_val = list()
                            
                with torch.no_grad():
                    for j, datapoint_val in enumerate(validation_loader):

                        # (See comments above on the training routine as it is almost the same)

                        padded_array_val = cnn_pad(datapoint_val[4].float(),2)
                        padded_array_val = torch.tensor(padded_array_val)

                        if device == "cuda":
                            datapoint_val[4] = datapoint_val[4].cuda()
                            padded_array_val = padded_array_val.cuda()
                        
                        if j%25 == 0:
                            print("Validation sample number",j+1,"/",len(validation_loader),"---",datapoint_val[0][0],"--- Input shape:",datapoint_val[4].shape)
                        
                        outputs = model(padded_array_val)

                        beat_activation_val = madmom.utils.quantize_events(datapoint_val[1][0], fps=100, length=len(datapoint_val[4][0]))
                        widen_beat_targets(beat_activation_val)
                        beat_activation_val = torch.tensor(beat_activation_val)

                        beat_activation_db_val = madmom.utils.quantize_events(datapoint_val[2][0], fps=100, length=len(datapoint_val[4][0]))
                        widen_beat_targets(beat_activation_db_val)
                        beat_activation_db_val = torch.tensor(beat_activation_db_val)

                        loss_val_b = criterion(outputs[0][:,0].float(),beat_activation_val.float().to(device))
                        loss_val_db = criterion(outputs[1][:,0].float(),beat_activation_db_val.float().to(device))

                        #Calculate F-Score (Beats):
                        try:
                            proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
                            beat_times = proc(outputs[0][:,0].detach().cpu().numpy())
                            evaluate = madmom.evaluation.beats.BeatEvaluation(beat_times, datapoint_val[1][0])
                            f_score_val = evaluate.fmeasure
                        except Exception as e:
                            print("Test sample cannot be processed correctly. Error in beat process:",e)
                            f_score_val = 0
                        fscore_list_val.append(f_score_val)

                        combined_0 = outputs[0].detach().cpu().numpy().squeeze()
                        combined_1 = outputs[1].detach().cpu().numpy().squeeze()
                        combined_act = np.vstack((np.maximum(combined_0 - combined_1, 0), combined_1)).T

                        #Calculate F-Score (Downbeats):
                        try:
                            proc_db = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[2,3,4],fps=100)
                            beat_times_db = proc_db(combined_act)
                            evaluate_db = madmom.evaluation.beats.BeatEvaluation(beat_times_db, datapoint_val[2][0], downbeats=True)
                            fscore_db_val = evaluate_db.fmeasure
                        except Exception as e:
                            print("Test sample cannot be processed correctly. Error in downbeat process:",e)
                            fscore_db_val = 0
                        fscore_list_db_val.append(fscore_db_val)

                        if j%25 == 0:
                            print("Loss (b):","%.4f"%loss_val_b.detach(),"--- Loss (db):","%.4f"%loss_val_db.detach(),"--- F-Score (b):","%.4f"%f_score_val,"--- F-Score (db):","%.4f"%fscore_db_val,"--- Average validation F-Score (b):","%.4f"%(np.sum(fscore_list_val)/len(fscore_list_val)),"\n")

                        loss_val = loss_val_b + loss_val_db

                        val_loss += loss_val.detach()
                        with torch.no_grad():
                            torch.cuda.empty_cache()
                
                val_loss_list.append(val_loss.cpu().item())
                fscore_list_val_average.append(np.sum(fscore_list_val)/len(fscore_list_val))
                fscore_list_val_average_db.append(np.sum(fscore_list_db_val)/len(fscore_list_db_val))
                learning_rates_list.append(optimizer.param_groups[0]['lr'])

                learning_rates = [0.001,0.0005,0.0001,0.00005,0.00001]

                # Adjust learning rate if validation loss increases:
                if len(val_loss_list) > 1:
                    if val_loss_list[-1] < val_loss_list[-2]:
                        print("Validation loss decreased. Learning rate stays at",optimizer.param_groups[0]['lr'],"\n")
                    else:
                        if optimizer.param_groups[0]['lr'] > learning_rates[-1]:
                            optimizer.param_groups[0]['lr'] = learning_rates[learning_rates.index(optimizer.param_groups[0]['lr'])+1]
                            print("Validation loss increased! Adjusted learning rate to",optimizer.param_groups[0]['lr'],"\n")
                        else:
                            print("Validation loss increased! But learning rate already reached minimum value...")

                # Plot validation results (history):
                plot_value_list(val_loss_list,"green","Validation summary loss after "+str(len(val_loss_list))+" validation runs",save=True,save_name=figures_dir+"Validation_loss_"+dataset)
                plot_value_list(fscore_list_val_average,"red","Validation F-Score (beats) after "+str(len(fscore_list_val_average))+" validation runs",save=True,save_name=figures_dir+"Validation_f_score_b_"+dataset)
                plot_value_list(fscore_list_val_average_db,"orange","Validation F-Score (downbeats) after "+str(len(fscore_list_val_average_db))+" validation runs",save=True,save_name=figures_dir+"Validation_f_score_db_"+dataset)
                
                print("Validation summary:\nIteration:",iter,"--- Validation Loss:","%.4f"%val_loss,"--- Average F-Score (b):","%.4f"%(np.sum(fscore_list_val)/len(fscore_list_val)),"--- Average F-Score (db):","%.4f"%(np.sum(fscore_list_db_val)/len(fscore_list_db_val)),"\n")

                # Save the model:
                total_time = (time.time()-begin_time) + loaded_time
                d,h,m,s = show_time(total_time)
                torch.save(model,checkpoints_dir+dataset+'/model_iter_'+str(iter)+'.pt')
                if sample_number / len(train_loader) > 0.7:
                    epoch_to_save = epoch+1
                else:
                    epoch_to_save = epoch+0
                # Create a dictionary with all necessary model progress statistics:
                save_dict = {"epoch":epoch_to_save,"iter":iter,"loss_list":loss_list,"val_loss_list":val_loss_list,"fscore_list_val_average":fscore_list_val_average,"fscore_list_val_average_db":fscore_list_val_average_db,"total_time":total_time,"learning_rate":optimizer.param_groups[0]['lr'],"learning_rates_list":learning_rates_list}
                with open(checkpoints_dir+dataset+'/model_iter_'+str(iter)+'_progress_info.json', 'w') as f:
                    json.dump(save_dict, f, indent=2)
                print("Model saved at iteration:",iter,"--- Total training time:",d,"days,",f"{h:02d}:{m:02d}:{s:02d}","\n")
            
        scheduler.step()
    """
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train ISMIR.')

    parser.add_argument('--checkpoints_dir', type=str, help='Where to store the results.')
    parser.add_argument('--figures_dir', type=str, help='Where to store the plots.')
    parser.add_argument('--dataset', type=str, help='Which dataset?')
    parser.add_argument('--mode', type=str, help='ismir/pm2s')
    parser.add_argument('--validate_every', type=str, help='when to validate')

    args = parser.parse_args()

    train(args)