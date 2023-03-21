import torch.nn as nn
import torch.nn.functional as F
import torch

import pytorch_lightning as pl

from beat_tracking.models import MyMadmom
from beat_tracking.helper_functions import *

import madmom

class MyMadmomModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MyMadmom()
        
    def forward(self,x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001,weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer],[scheduler]
    
    def configure_callbacks(self):
        pass
    
    def training_step(self,batch,batch_idx):
        criterion = torch.nn.BCELoss()
        file,beats,downbeats,idx,pr = batch
        padded_array = cnn_pad(pr.float(),2)
        padded_array = torch.tensor(padded_array)
        outputs = self(padded_array)
        # Get beat activation function from the time annotations and widen beat targets for better accuracy:                
        beat_activation = madmom.utils.quantize_events(beats[0].cpu(), fps=100, length=len(pr[0]))
        widen_beat_targets(beat_activation)
        beat_activation = torch.tensor(beat_activation).cuda()

        # Same for downbeats:
        beat_activation_db = madmom.utils.quantize_events(downbeats[0].cpu(), fps=100, length=len(pr[0]))
        widen_beat_targets(beat_activation_db)
        beat_activation_db = torch.tensor(beat_activation_db).cuda()
        
        loss_b = criterion(outputs[0][:,0].float(),beat_activation.float())
        loss_db = criterion(outputs[1][:,0].float(),beat_activation_db.float())
        loss = loss_b + loss_db
        
        logs = {
            'train_loss': loss,
            'train_loss_b': loss_b,
            'train_loss_db': loss_db,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}
    
    def validation_step(self,batch,batch_idx):
        criterion = torch.nn.BCELoss()
        file,beats,downbeats,idx,pr = batch
        padded_array = cnn_pad(pr.float(),2)
        padded_array = torch.tensor(padded_array)
        outputs = self(padded_array)
        
        # Get beat activation function from the time annotations and widen beat targets for better accuracy:                
        beat_activation = madmom.utils.quantize_events(beats[0].cpu(), fps=100, length=len(pr[0]))
        widen_beat_targets(beat_activation)
        beat_activation = torch.tensor(beat_activation).cuda()

        # Same for downbeats:
        beat_activation_db = madmom.utils.quantize_events(downbeats[0].cpu(), fps=100, length=len(pr[0]))
        widen_beat_targets(beat_activation_db)
        beat_activation_db = torch.tensor(beat_activation_db).cuda()
        
        loss_b = criterion(outputs[0][:,0].float(),beat_activation.float())
        loss_db = criterion(outputs[1][:,0].float(),beat_activation_db.float())
        loss = loss_b + loss_db
        
        #Calculate F-Score (Beats):
        try:
            proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
            beat_times = proc(outputs[0][:,0].detach().cpu().numpy())
            evaluate = madmom.evaluation.beats.BeatEvaluation(beat_times, beats[0].cpu())
            f_score_val = evaluate.fmeasure
        except Exception as e:
            print("Test sample cannot be processed correctly. Error in beat process:",e)
            f_score_val = 0
            
        combined_0 = outputs[0].detach().cpu().numpy().squeeze()
        combined_1 = outputs[1].detach().cpu().numpy().squeeze()
        combined_act = np.vstack((np.maximum(combined_0 - combined_1, 0), combined_1)).T

        #Calculate F-Score (Downbeats):
        try:
            proc_db = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[2,3,4],fps=100)
            beat_times_db = proc_db(combined_act)
            evaluate_db = madmom.evaluation.beats.BeatEvaluation(beat_times_db, downbeats[0].cpu(), downbeats=True)
            fscore_db_val = evaluate_db.fmeasure
        except Exception as e:
            print("Test sample cannot be processed correctly. Error in downbeat process:",e)
            fscore_db_val = 0
        
        logs = {
            'val_loss': loss,
            'val_loss_b': loss_b,
            'val_loss_db': loss_db,
            'f_score_b': f_score_val,
            'f_score_db': fscore_db_val,
        }
        self.log_dict(logs, prog_bar=True)
        
        return {'val_loss': loss, 'logs': logs}