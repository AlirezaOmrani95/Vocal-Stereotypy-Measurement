import torch
from torch import nn
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryCohenKappa 
from torch.utils.tensorboard import SummaryWriter
from utils import Audio_Dataset as AD, train_valid_separation, save_checkpoint, create_data_loader
from tqdm import tqdm
import numpy as np
from datetime import datetime
import os
from pretrained_models import Pretrained_Models
import random
import argparse

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def arg_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_dir', default= f'runs/BP_{timestamp}', help= 'The address that logs will be saved')
    parser.add_argument('--dataset_dir', default= './dataset/train/', help= 'The address to read data from')
    parser.add_argument('--annotation_dir',default= './Annotation_file.csv', help= 'The location of input CSV file')
    parser.add_argument('--batch_size', default= 128)
    parser.add_argument('--epoch_number', default= 25)
    parser.add_argument('--lr_rate', default= 1e-4)
    parser.add_argument('--num_classes', default= 2)
    parser.add_argument('--pretrained_dir', default= './weights/pretrained weight/pretrained weight', help='The location of the pretrained weight file')
    return(parser.parse_args())


def train(model,train_data_loader,valid_data_loader,loss_fn, metrics,optimizer,device,epochs,writer):
    best_kappa = 0
    for epoch in range(epochs):
        model.train(True)
        train_hisotry = train_single_epoch(model,train_data_loader,loss_fn, metrics,optimizer,device,epoch, writer)
        model.eval()
        valid_history = validation_single_epoch(model,valid_data_loader,loss_fn,metrics,device,epoch,writer)
        
        writer.add_scalars("Training vs. Validation Loss",{"Training":train_hisotry["loss"],"Validation":valid_history["loss"]},epoch+1)
        writer.add_scalars("Training vs. Validation Accuracy", {"Training":train_hisotry["b_acc"],"Validation":valid_history["b_acc"]},epoch+1)
        writer.add_scalars("Training vs. Validation F1_score", {"Training":train_hisotry["b_f1"],"Validation":valid_history["b_f1"]},epoch+1)
        writer.add_scalars("Training vs. Validation Kappa_Cohen", {"Training":train_hisotry["b_kappa"],"Validation":valid_history["b_kappa"]},epoch+1)
        if valid_history['b_kappa'] > best_kappa:
            best_kappa = valid_history['b_kappa']
            if not os.path.exists(model_weight_path):
                os.mkdir(model_weight_path)
            model_path = f"{model_weight_path}/model_{timestamp}_{epoch}"
            save_checkpoint(model,optimizer,model_path,epoch)
        elif epoch == epochs-1:
            model_path = f"{model_weight_path}/model_{timestamp}_{epoch}"
            save_checkpoint(model,optimizer,model_path,epoch)

def train_single_epoch(model, data_loader, loss_fn, metrics,optimizer, device,epoch, writer):
    loss_lst = []
    b_acc_lst = []
    b_f1_lst = []
    b_kappa_lst = []
    with tqdm(data_loader,unit='batch') as t_data_loader:
        for counter,(input_, target_) in enumerate(t_data_loader):
            t_data_loader.set_description(f'Epoch_Train {epoch+1}')
            optimizer.zero_grad()
                
            input_,target_ = input_.to(device), torch.unsqueeze(target_,-1).to(device)
                
            pred = model(input_)
                
            loss = loss_fn(pred,target_.float())
            loss.backward()

            b_acc = metrics[0](torch.sigmoid(pred),target_)
            b_f1 = metrics[1](torch.sigmoid(pred),target_)
            b_kappa = metrics[2](torch.sigmoid(pred),target_)
            optimizer.step()

            loss_lst.append(loss.item())
            b_acc_lst.append(b_acc.item())
            b_f1_lst.append(b_f1.item())
            b_kappa_lst.append(b_kappa.item())

            t_data_loader.set_postfix(loss=np.mean(loss_lst), binary_accuracy = np.mean(b_acc_lst), binary_f1 = np.mean(b_f1_lst),binary_kappa = np.mean(b_kappa_lst))
            if counter % 50 == 0:
                tbx = epoch * len(data_loader) + counter + 1
                writer.add_scalar("Loss/train",np.mean(loss_lst),tbx)
                writer.add_scalar("Accuracy/train",np.mean(b_acc_lst),tbx)
                writer.add_scalar("F1_score/train",np.mean(b_f1_lst),tbx)
                writer.add_scalar("Kappa_Cohen/train",np.mean(b_kappa_lst),tbx)
                
    history = {'loss':np.mean(loss_lst),'b_acc':np.mean(b_acc_lst),
               'b_f1':np.mean(b_f1_lst),'b_kappa':np.mean(b_kappa_lst)}
    return history
  
def validation_single_epoch(model, data_loader, loss_fn, metrics, device,epoch,writer):
    loss_lst = []
    b_acc_lst = []
    b_f1_lst = []
    b_kappa_lst = []
    with tqdm(data_loader,unit='batch') as t_data_loader:
        with torch.no_grad():
            for counter, (input_, target_) in enumerate(t_data_loader):
                t_data_loader.set_description(f'Epoch_Valid {epoch+1}')
                
                input_,target_ = input_.to(device), torch.unsqueeze(target_,-1).to(device)
                
                pred = model(input_)
                
                loss = loss_fn(pred,target_.float())
                
                b_acc = metrics[0](torch.sigmoid(pred),target_)
                b_f1 = metrics[1](torch.sigmoid(pred),target_)
                b_kappa = metrics[2](torch.sigmoid(pred),target_)

                loss_lst.append(loss.item())
                b_acc_lst.append(b_acc.item())
                b_f1_lst.append(b_f1.item())
                b_kappa_lst.append(b_kappa.item())

                t_data_loader.set_postfix(loss=np.mean(loss_lst), binary_accuracy = np.mean(b_acc_lst), binary_f1 = np.mean(b_f1_lst),binary_kappa = np.mean(b_kappa_lst))

                if counter % 50 == 0:
                    tbx = epoch * len(data_loader) + counter + 1
                    writer.add_scalar("Loss/valid",np.mean(loss_lst),tbx)
                    writer.add_scalar("Accuracy/valid",np.mean(b_acc_lst),tbx)
                    writer.add_scalar("F1_score/valid",np.mean(b_f1_lst),tbx)
                    writer.add_scalar("Kappa_Cohen/valid",np.mean(b_kappa_lst),tbx)

    history = {'loss':np.mean(loss_lst),'b_acc':np.mean(b_acc_lst),
               'b_f1':np.mean(b_f1_lst),'b_kappa':np.mean(b_kappa_lst)}
    return history

if __name__ == '__main__':
    #General Info
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    args = arg_parsing()
     
    model_weight_path = os.path.join(args.logs_dir,'model_weight')

    writer = SummaryWriter(args.logs_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Implementing the audio data loader
    train_annotations, valid_annotations = train_valid_separation(args.annotation_dir)
    
    train_dataset = AD(args.dataset_dir,train_annotations, device=device)
    valid_dataset = AD(args.dataset_dir,valid_annotations, device=device)

    train_data_loader = create_data_loader(train_dataset,args.batch_size)
    valid_data_loader = create_data_loader(valid_dataset,1024)

    #loss function and accuracy

    #In case of having imbalance data, you can use weighted loss function.
    weights = torch.tensor(train_dataset._get_class_num_(), dtype= torch.float32)
    weights = weights.sum() / (weights * 2)
    criterion = nn.BCEWithLogitsLoss(pos_weight= weights[-1]).to(device)

    metrics = []
    metrics.append(BinaryAccuracy().to(device))
    metrics.append(BinaryF1Score().to(device))
    metrics.append(BinaryCohenKappa().to(device))

    #Implementing the model
    model = Pretrained_Models('xcit_tiny_12_p8_224',10,(2,64,44)).get_model()
    
    model = model.to(device)
    model.load_state_dict(torch.load(args.pretrained_dir)['model_state_dict'])
    for param in model.parameters():
        param.requires_grad == False

    model.head = nn.Linear(model.head.in_features,1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr_rate)
    
    train(model,train_data_loader, valid_data_loader,criterion, metrics,optimizer,device,args.epoch_number,writer)
    writer.flush()
