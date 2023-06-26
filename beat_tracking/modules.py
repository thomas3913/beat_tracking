import torch.nn as nn
import torch.nn.functional as F
import torch

import pytorch_lightning as pl
import matplotlib.pyplot as plt

from beat_tracking.models import MyMadmom, RNNJointBeatModel
from beat_tracking.postprocessing import RNNJointBeatProcessor, get_beat_activation_function_from_probs
from beat_tracking.helper_functions import *

import madmom

def configure_optimizers_pm2s(module, lr=1e-3, step_size=50):
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=lr,
        betas=(0.8, 0.8),
        eps=1e-4,
        weight_decay=1e-2,
    )
    scheduler_lrdecay = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=0.1
    )
    return [optimizer], [scheduler_lrdecay]

def configure_callbacks(monitor='f_score_b', mode='max'):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        filename='{epoch}-{val_loss:.2f}-{val_f1:.2f}',
        save_last=False,
    )
    earlystop_callback = pl.callbacks.EarlyStopping(
        monitor=monitor,
        patience=200,
        mode=mode,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    
    return [checkpoint_callback, earlystop_callback, lr_monitor]


def f_measure_framewise(y, y_hat):
    acc = (y_hat == y).float().mean()
    TP = torch.logical_and(y_hat==1, y==1).float().sum()
    FP = torch.logical_and(y_hat==1, y==0).float().sum()
    FN = torch.logical_and(y_hat==0, y==1).float().sum()

    p = TP / (TP + FP + np.finfo(float).eps)
    r = TP / (TP + FN + np.finfo(float).eps)
    f = 2 * p * r / (p + r + np.finfo(float).eps)
    return acc, p, r, f


class AudioModule(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.model = MyMadmom()
        self.stepsize = 15
        
    def forward(self,x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001,weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.stepsize, gamma=0.1)
        return [optimizer],[scheduler]
    
    def configure_callbacks(self):
        return configure_callbacks(monitor='val_f_b')
    
    def training_step(self,batch,batch_idx):

        criterion = torch.nn.BCELoss()
        #file,beats,downbeats,idx,pr = batch
        _, beats, downbeats, spectrogram, _, _ = batch
        if spectrogram.shape[1] > 100000:
            return None

        padded_array = cnn_pad(spectrogram,2)
        padded_array = torch.tensor(padded_array)

        outputs = self(padded_array)
        # Get beat activation function from the time annotations and widen beat targets for better accuracy:                
        beat_activation = madmom.utils.quantize_events(beats[0].cpu(), fps=100, length=len(spectrogram[0]))
        widen_beat_targets(beat_activation)
        beat_activation = torch.tensor(beat_activation).cuda()

        # Same for downbeats:
        beat_activation_db = madmom.utils.quantize_events(downbeats[0].cpu(), fps=100, length=len(spectrogram[0]))
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
        #file,beats,downbeats,idx,pr = batch
        _, beats, downbeats, spectrogram, _, _ = batch
        if spectrogram.shape[1] > 100000:
            return None
        padded_array = cnn_pad(spectrogram,2)
        padded_array = torch.tensor(padded_array)

        outputs = self(padded_array)        
        
        # Get beat activation function from the time annotations and widen beat targets for better accuracy:                
        beat_activation = madmom.utils.quantize_events(beats[0].cpu(), fps=100, length=len(spectrogram[0]))
        widen_beat_targets(beat_activation)
        beat_activation = torch.tensor(beat_activation).cuda()

        # Same for downbeats:
        beat_activation_db = madmom.utils.quantize_events(downbeats[0].cpu(), fps=100, length=len(spectrogram[0]))
        widen_beat_targets(beat_activation_db)
        beat_activation_db = torch.tensor(beat_activation_db).cuda()
        
        loss_b = criterion(outputs[0][:,0].float(),beat_activation.float())
        loss_db = criterion(outputs[1][:,0].float(),beat_activation_db.float())

        #Calculate F-Score (Beats):
        try:
            proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
            beat_times = proc(outputs[0][:,0].detach().cpu().numpy())
            evaluate = madmom.evaluation.beats.BeatEvaluation(beat_times, beats[0].cpu())
            f_score_val = evaluate.fmeasure
        except Exception as e:
            print("Test sample cannot be processed correctly. Error in beat process:",e)
            f_score_val = 0
            
        #Calculate F-Score (Downbeats):
        loss = loss_b + loss_db

        combined_0 = outputs[0].detach().cpu().numpy().squeeze()
        combined_1 = outputs[1].detach().cpu().numpy().squeeze()
        combined_act = np.vstack((np.maximum(combined_0 - combined_1, 0), combined_1)).T

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
            'val_f_b': f_score_val,
            'val_f_db': fscore_db_val,
        }

        self.log_dict(logs, prog_bar=True)
        
        return {'val_loss': loss, 'logs': logs}
    
    def test_step(self,batch,batch_idx):

        #file,beats,downbeats,idx,pr = batch
        _, beats, downbeats, spectrogram, _, _ = batch
        if spectrogram.shape[1] > 100000:
            return None

        padded_array = cnn_pad(spectrogram,2)
        padded_array = torch.tensor(padded_array)

        outputs = self(padded_array)
        
        baf = outputs[0][:,0].detach().cpu().numpy()
        #plt.clf()
        #plt.plot(np.arange(1000),baf[:1000])
        #plt.savefig("temp_fig.png")

        #Calculate F-Score (Beats):
        try:
            proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
            beat_times = proc(outputs[0][:,0].detach().cpu().numpy())
            evaluate = madmom.evaluation.beats.BeatEvaluation(beat_times, beats[0].cpu())
            f_score_test = evaluate.fmeasure
        except Exception as e:
            print("Test sample cannot be processed correctly. Error in beat process:",e)
            f_score_test = 0
            
        #Calculate F-Score (Downbeats):

        combined_0 = outputs[0].detach().cpu().numpy().squeeze()
        combined_1 = outputs[1].detach().cpu().numpy().squeeze()
        combined_act = np.vstack((np.maximum(combined_0 - combined_1, 0), combined_1)).T

        try:
            proc_db = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[2,3,4],fps=100)
            beat_times_db = proc_db(combined_act)
            evaluate_db = madmom.evaluation.beats.BeatEvaluation(beat_times_db, downbeats[0].cpu(), downbeats=True)
            fscore_db_test = evaluate_db.fmeasure
        except Exception as e:
            print("Test sample cannot be processed correctly. Error in downbeat process:",e)
            fscore_db_test = 0
        
        logs = {
            'f_score_b_test': f_score_test,
            'f_score_db_test': fscore_db_test,
        }

        self.log_dict(logs, prog_bar=True)
        
        return {'test_loss': 0, 'logs': logs}



class MyMadmomModule(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.model = MyMadmom()
        self.stepsize = args.stepsize
        
    def forward(self,x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001,weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.stepsize, gamma=0.1)
        return [optimizer],[scheduler]
    
    def configure_callbacks(self):
        return configure_callbacks(monitor='f_score_b')
    
    def training_step(self,batch,batch_idx):

        criterion = torch.nn.BCELoss()
        #file,beats,downbeats,idx,pr = batch
        _,_,_,_,_,beats,downbeats,pr = batch
        if pr.shape[1] > 100000:
            return None

        padded_array = cnn_pad(pr.float(),2)
        padded_array = torch.tensor(padded_array)

        if padded_array.shape[2] > 88:
            padded_array = padded_array[:,:,21:109]

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
        #file,beats,downbeats,idx,pr = batch
        _,_,_,_,_,beats,downbeats,pr = batch
        if pr.shape[1] > 100000:
            return None
        padded_array = cnn_pad(pr.float(),2)
        padded_array = torch.tensor(padded_array)

        if padded_array.shape[2] > 88:
            padded_array = padded_array[:,:,21:109]

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

        #Calculate F-Score (Beats):
        try:
            proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
            beat_times = proc(outputs[0][:,0].detach().cpu().numpy())
            evaluate = madmom.evaluation.beats.BeatEvaluation(beat_times, beats[0].cpu())
            f_score_val = evaluate.fmeasure
        except Exception as e:
            print("Test sample cannot be processed correctly. Error in beat process:",e)
            f_score_val = 0
            
        #Calculate F-Score (Downbeats):
        loss = loss_b + loss_db

        combined_0 = outputs[0].detach().cpu().numpy().squeeze()
        combined_1 = outputs[1].detach().cpu().numpy().squeeze()
        combined_act = np.vstack((np.maximum(combined_0 - combined_1, 0), combined_1)).T

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
            'val_f_b': f_score_val,
            'val_f_db': fscore_db_val,
        }

        self.log_dict(logs, prog_bar=True)
        
        return {'val_loss': loss, 'logs': logs}
    
    def test_step(self,batch,batch_idx):

        #file,beats,downbeats,idx,pr = batch
        _,_,_,_,_,beats,downbeats,pr = batch
        if pr.shape[1] > 100000:
            return None
        padded_array = cnn_pad(pr.float(),2)
        padded_array = torch.tensor(padded_array)

        if padded_array.shape[2] > 88:
            padded_array = padded_array[:,:,21:109]

        outputs = self(padded_array)
        
        baf = outputs[0][:,0].detach().cpu().numpy()
        #plt.clf()
        #plt.plot(np.arange(1000),baf[:1000])
        #plt.savefig("temp_fig.png")

        #Calculate F-Score (Beats):
        try:
            proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
            beat_times = proc(outputs[0][:,0].detach().cpu().numpy())
            evaluate = madmom.evaluation.beats.BeatEvaluation(beat_times, beats[0].cpu())
            f_score_test = evaluate.fmeasure
        except Exception as e:
            print("Test sample cannot be processed correctly. Error in beat process:",e)
            f_score_test = 0
            
        #Calculate F-Score (Downbeats):

        combined_0 = outputs[0].detach().cpu().numpy().squeeze()
        combined_1 = outputs[1].detach().cpu().numpy().squeeze()
        combined_act = np.vstack((np.maximum(combined_0 - combined_1, 0), combined_1)).T

        try:
            proc_db = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[2,3,4],fps=100)
            beat_times_db = proc_db(combined_act)
            evaluate_db = madmom.evaluation.beats.BeatEvaluation(beat_times_db, downbeats[0].cpu(), downbeats=True)
            fscore_db_test = evaluate_db.fmeasure
        except Exception as e:
            print("Test sample cannot be processed correctly. Error in downbeat process:",e)
            fscore_db_test = 0
        
        logs = {
            'f_score_b_test': f_score_test,
            'f_score_db_test': fscore_db_test,
        }

        self.log_dict(logs, prog_bar=True)
        
        return {'test_loss': 0, 'logs': logs}


class BeatModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = RNNJointBeatModel()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return configure_optimizers_pm2s(self)

    def configure_callbacks(self):
        return configure_callbacks(monitor='val_f_b')

    def training_step(self, batch, batch_idx):
        # Data
        x, y_b, y_db, y_ibi, length, _, _, _ = batch
        x = x.float()
        y_b = y_b.float()
        y_db = y_db.float()
        y_ibi = y_ibi.long()
        length = length.long()

        # Forward pass
        y_b_hat, y_db_hat, y_ibi_hat = self(x)

        # Mask out the padding part
        mask = torch.ones(y_b_hat.shape).to(y_b_hat.device)
        for i in range(y_b_hat.shape[0]):
            mask[i, length[i]:] = 0
        y_b_hat = y_b_hat * mask
        y_db_hat = y_db_hat * mask
        y_ibi_hat = y_ibi_hat * mask.unsqueeze(1)

        # Loss
        loss_b = F.binary_cross_entropy(y_b_hat, y_b)
        loss_db = F.binary_cross_entropy(y_db_hat, y_db)
        loss_ibi = nn.NLLLoss(ignore_index=0)(y_ibi_hat, y_ibi)
        loss = loss_b + loss_db + loss_ibi

        # Logging
        logs = {
            'train_loss': loss,
            'train_loss_b': loss_b,
            'train_loss_db': loss_db,
            'train_loss_ibi': loss_ibi,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}

    def validation_step(self, batch, batch_idx):
        # Data
        x, y_b, y_db, y_ibi, length, _, _ ,_= batch
        x = x.float()
        y_b = y_b.float()
        y_db = y_db.float()
        y_ibi = y_ibi.long()
        length = length.long()

        # Forward pass
        y_b_hat, y_db_hat, y_ibi_hat = self(x)

        # Mask out the padding part
        for i in range(y_b_hat.shape[0]):
            y_b_hat[i, length[i]:] = 0
            y_db_hat[i, length[i]:] = 0
            y_ibi_hat[i, :, length[i]:] = 0

        # Loss
        loss_b = F.binary_cross_entropy(y_b_hat, y_b)
        loss_db = F.binary_cross_entropy(y_db_hat, y_db)
        loss_ibi = nn.NLLLoss(ignore_index=0)(y_ibi_hat, y_ibi)
        loss = loss_b + loss_db + loss_ibi

        # Metrics
        accs_b, precs_b, recs_b, fs_b = 0, 0, 0, 0
        accs_db, precs_db, recs_db, fs_db = 0, 0, 0, 0

        for i in range(x.shape[0]):
            # get sample from batch
            y_b_hat_i = torch.round(y_b_hat[i, :length[i]])
            y_db_hat_i = torch.round(y_db_hat[i, :length[i]])
            y_ibi_hat_i = y_ibi_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            
            y_b_i = y_b[i, :length[i]]
            y_db_i = y_db[i, :length[i]]
            y_ibi_i = y_ibi[i, :length[i]]

            # filter out ignore indexes
            y_ibi_hat_i = y_ibi_hat_i[y_ibi_i != 0]
            y_ibi_i = y_ibi_i[y_ibi_i != 0]

            # get accuracy

            acc_b, prec_b, rec_b, f_b = f_measure_framewise(y_b_i, y_b_hat_i)
            acc_db, prec_db, rec_db, f_db = f_measure_framewise(y_db_i, y_db_hat_i)
            
            accs_b += acc_b
            precs_b += prec_b
            recs_b += rec_b
            fs_b += f_b

            accs_db += acc_db
            precs_db += prec_db
            recs_db += rec_db
            fs_db += f_db

        accs_b /= x.shape[0]
        precs_b /= x.shape[0]
        recs_b /= x.shape[0]
        fs_b /= x.shape[0]

        accs_db /= x.shape[0]
        precs_db /= x.shape[0]
        recs_db /= x.shape[0]
        fs_db /= x.shape[0]

        # Logging
        logs = {
            'val_loss': loss,
            #'val_loss_b': loss_b,
            #'val_loss_db': loss_db,
            #'val_loss_ibi': loss_ibi,
            'val_acc_b': accs_b,
            'val_prec_b': precs_b,
            #'val_rec_b': recs_b,
            'val_f_b': fs_b,
            'val_acc_db': accs_db,
            'val_prec_db': precs_db,
            #'val_rec_db': recs_db,
            'val_f_db': fs_db,
            #'val_f1': fs_b,  # this will be used as the monitor for logging and checkpointing callbacks
        }
        self.log_dict(logs, prog_bar=True)

        return {'val_loss': loss, 'logs': logs}
    
    def test_step(self,batch,batch_idx):
        # Data
        x, y_b, y_db, y_ibi, length, y_beats, y_downbeats,_ = batch
        x = x.float()
        y_b = y_b.float()
        y_db = y_db.float()
        y_ibi = y_ibi.long()
        length = length.long()

        # Forward pass
        y_b_hat, y_db_hat, y_ibi_hat = self(x)

        # Mask out the padding part
        for i in range(y_b_hat.shape[0]):
            y_b_hat[i, length[i]:] = 0
            y_db_hat[i, length[i]:] = 0
            y_ibi_hat[i, :, length[i]:] = 0

        # Metrics
        accs_b, precs_b, recs_b, fs_b = 0, 0, 0, 0
        accs_db, precs_db, recs_db, fs_db = 0, 0, 0, 0

        for i in range(x.shape[0]):
            # get sample from batch
            y_b_hat_i = torch.round(y_b_hat[i, :length[i]])
            y_db_hat_i = torch.round(y_db_hat[i, :length[i]])
            y_ibi_hat_i = y_ibi_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            
            y_b_i = y_b[i, :length[i]]
            y_db_i = y_db[i, :length[i]]
            y_ibi_i = y_ibi[i, :length[i]]

            # filter out ignore indexes
            y_ibi_hat_i = y_ibi_hat_i[y_ibi_i != 0]
            y_ibi_i = y_ibi_i[y_ibi_i != 0]

            # get accuracy

            acc_b, prec_b, rec_b, f_b = f_measure_framewise(y_b_i, y_b_hat_i)
            acc_db, prec_db, rec_db, f_db = f_measure_framewise(y_db_i, y_db_hat_i)
            
            accs_b += acc_b
            precs_b += prec_b
            recs_b += rec_b
            fs_b += f_b

            accs_db += acc_db
            precs_db += prec_db
            recs_db += rec_db
            fs_db += f_db

        accs_b /= x.shape[0]
        precs_b /= x.shape[0]
        recs_b /= x.shape[0]
        fs_b /= x.shape[0]

        accs_db /= x.shape[0]
        precs_db /= x.shape[0]
        recs_db /= x.shape[0]
        fs_db /= x.shape[0]

        #Post processing with RNNJointBeatProcessor:
        post_processor = RNNJointBeatProcessor()
        beats = post_processor.process(x,y_b_hat,y_db_hat)
        evaluate = madmom.evaluation.beats.BeatEvaluation(y_beats[0].detach().cpu(), beats)
        f_b_post = evaluate.fmeasure
            
        # Post processing with DBN:
        beat_activation_function, beat_activation_function_modified = get_beat_activation_function_from_probs(x,y_b_hat,y_db_hat)
        proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100,min_bpm=55.0, max_bpm=215.0, transition_lambda=100, threshold=0.05)
        assert len(beat_activation_function) > 0
        beats_DBN = proc(beat_activation_function)
        
        proc_2 = madmom.features.beats.DBNBeatTrackingProcessor(fps=100,min_bpm=55.0, max_bpm=215.0, transition_lambda=100, threshold=0.05)
        beats_DBN_modified = proc_2(beat_activation_function_modified)
        
        #plt.clf()
        #plt.plot(np.arange(1000),beat_activation_function[:1000])
        #plt.savefig("temp_fig_2.png")
        
        evaluate_2 = madmom.evaluation.beats.BeatEvaluation(y_beats[0].detach().cpu(), beats_DBN)
        f_b_post_DBN = evaluate_2.fmeasure
        
        evaluate_2_modified = madmom.evaluation.beats.BeatEvaluation(y_beats[0].detach().cpu(), beats_DBN_modified)
        f_b_post_DBN_modified = evaluate_2_modified.fmeasure

        # Logging
        logs = {
            'test_acc_b': accs_b,
            'test_prec_b': precs_b,
            #'test_rec_b': recs_b,
            'test_f_b': fs_b,
            'test_acc_db': accs_db,
            'test_prec_db': precs_db,
            #'test_rec_db': recs_db,
            'test_f_db': fs_db,
            'f_b_postprocessing':f_b_post,
            'f_b_postprocessing_DBN':f_b_post_DBN,
            'f_b_postprocessing_DBN_modified':f_b_post_DBN_modified,
            #'test_f1': fs_b,  # this will be used as the monitor for logging and checkpointing callbacks
        }
        self.log_dict(logs, prog_bar=True)

        return {'test_loss': 0, 'logs': logs}
