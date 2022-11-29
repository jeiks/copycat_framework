import torch
import torch.nn as nn
import torch.optim as optim
#system
from os import path as os_path, listdir as os_listdir, rename as os_rename
#utils
from .data import Dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score,\
                            confusion_matrix, classification_report
import random
import cv2

def set_seeds(seed):
    '''
    from: <https://lifesaver.codes/answer/can-t-reproduce-results-even-set-all-random-seeds-7068> Date: 04/2022
    Here are some points to check:
    * https://pytorch.org/docs/stable/notes/randomness.html: Pytorch is not reproducible between CPU and GPU, between different platforms, and different commits.
    * Call this before running any of your functions:
        (you can put it in the beginning of your main code, right after importing your modules.
         If you do not use a module among these in your main code, it is ok, just import it and seed it so when you import it a second time,
         Python will find that it has already been imported so there will be nothing to do. Example: torch):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    * Following the above, you need to seed EVERY external module (outside Pytorch) that may introduce randomness in your entire code.
    * You need to set the init function of the worker(s) to be fed to the DataLoader:
        def _init_fn(worker_id):
          np.random.seed(int(seed))
      Make sure that your dataloader loads samples in the same order every call. If you do cropping, or other preprocessing steps,
      make sure that they are deterministic. As I remember, such modules can provide a deterministic result (in Pytorch 1.0.0 or even before this version).

    * I learned this recently, despite it was written in Pytoch doc the whole time!!!!: Unfortunately, some Pytorch modules follow non-deterministic behavior;
      and most of the time you can not get rid of it by following the above steps.
      Example: the BACKWARD of upsampling and interpolation functionals/classes is non-deterministic (see here: 
      <https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842?u=sbelharbi>).
      This means, if you use such modules in the training graph, you will never obtain a deterministic results no matter what you do.
      torch.nn.ConvTranspose2d is not deterministic unless you set torch.backends.cudnn.deterministic = True
      (they said you can try to make the operation deterministic ... by setting torch.backends.cudnn.deterministic = True.
      So I am not sure if by doing so, you certainly obtain a deterministic result).
      So, make sure you are not using any of the non-deterministic Pytorch modules.
    
    Print your intermediate results to find out from where the non-deterministic behavior comes from: loaded samples, crops, losses, ...

    In my experience, following the above steps lead to exactly same results.
    If it is not your case, you may be using an external library that is a source of randomness, or you are using a non-deterministic
    Pytorch module (or something else that I am not aware of it). If reproducibility is important for you (and I assume it is), you can check step by step
    your code to trace the source of randomness (run your code gradually, and inspect the results at every step).
    Make sure that the part of your code that you think it is deterministic is indeed deterministic.
    Let me know how it goes, and please, let me know what was the source of randomness in your case. I am curious. Thanks!
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cv2.setRNGSeed(seed)

def calculate_mean_std(dataset: Dataset, verbose=False):
    if verbose:
        try: from tqdm import tqdm
        except: tqdm = lambda x: x
    m = torch.tensor([.0,.0,.0])
    s = torch.tensor([.0,.0,.0])
    if verbose: target = tqdm(dataset.train)
    else:       target = dataset.train
    for img, _ in target:
        m += img.mean(dim=(1,2))
        s += img.std(dim=(1,2))
    l = len(dataset.train)
    return m/l, s/l

def check_file_and_rename(filename):
        if os_path.isfile(filename):
            numbers  = [-1]
            numbers += [int(x.split('.')[-1]) for x in os_listdir() if x.startswith(f'{filename}.')]
            os_rename(filename, f'{filename}.{max(numbers)+1}')

def get_copycat_model(model):
    #Checking if it is a Copycat's model or a Torch's model:
    if str(model).startswith('Copycat Model'):
        return model()
    elif str(model).startswith('<copycat.oracle') or str(model).startswith('<copycat.copycat'):
        return model.model()
    else:
        return model

def train_epoch(epoch, max_epochs, model, criterion, optimizer, device, dataset, db_name='train', batch_size=16, balance_dataset=False):
    model   = get_copycat_model(model)
    model   = model.to(device)
    db      = getattr(dataset, db_name)
    db.balance_dataset(balance_dataset)
    loader  = getattr(dataset, f'{db_name}_loader')
    display = len(db) // batch_size // (100 if len(db) < 1e5 else 1000) + 1 #the sum with 1 is to avoid the result zero on small datasets
    
    running_loss = 0.0
    total_loss   = 0.0
    correct      = 0
    n_preds      = 0
    model.train()
    with tqdm(loader(batch_size=batch_size)) as tqdm_train:
        for i, data in enumerate(tqdm_train):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            ## statistics
            # loss:
            total_loss  += loss.item()
            running_loss += loss.item()
            # correct predictions:
            predicted  = outputs.data.max(dim=1)[1]
            correct   += (predicted == labels).sum().item()
            n_preds   += labels.size(0)

            if i%display == display-1:
                tqdm_train.set_description('[ ' 
                                           f'Epoch: {epoch+1:02d}/{max_epochs} '
                                           f'Loss: {running_loss/display:.6f} '
                                           f'Accuracy: {correct/n_preds*100.:.2f}% '
                                           f'lr: {optimizer.param_groups[0]["lr"]:.2e}'
                                           ' ]')
                running_loss = 0.0
        tqdm_train.set_description(f'Epoch: {epoch+1}/{max_epochs} '
                                   f'Total Loss: {total_loss/n_preds:.6f} '
                                   f'Accuracy: {correct/n_preds*100.:.2f}% ')
    
    return model

def train(model, dataset, db_name='train',
                 max_epochs=10, batch_size=16,
                 criterion='CrossEntropyLoss', optimizer='SGD', lr=1e-3, weight_decay=True,
                 validation_step=0, db_name_validate='test',
                 snapshot_prefix=None, gamma=0.1,
                 balance_dataset=False):
    model = get_copycat_model(model)

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    criterion = getattr(nn, criterion)()
    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.5)
    
    if weight_decay:
        sch_1 = max_epochs//2           # time 1. Ex: max_epochs=20, time 1 = 10
        sch_2 = sch_1+max_epochs//3     # time 2.                    time 2 = 10 + 6 = 16
        sch_3 = sch_2+max_epochs//3//2  # time 3.                    time 3 = 16 + 3 = 19
        sch = np.array(sorted(list(set([sch_1, sch_2, sch_3]))))
        if len(sch[sch>0]) > 0:
            print(f"Scheduler Milestones: {sch.tolist()}")
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=sch.tolist(), gamma=gamma)
        else:
            print(f"It was not possible to define the milestones for scheduler (max_epochs={max_epochs} is too small).")
            weight_decay = False

    for epoch in range(max_epochs):
        model = train_epoch(epoch, max_epochs, model, criterion, optimizer, device, dataset, db_name, batch_size, balance_dataset)

        if weight_decay: scheduler.step()
        
        if validation_step != 0:
            if epoch % validation_step == 0:
                micro, macro = test(model, dataset, db_name=db_name_validate, batch_size=batch_size, metric='f1_score')
                print(f'~~ F1 Score on {db_name_validate} dataset:')
                print(f'~~ Micro Avg: {micro:.6f} Macro Avg: {macro:.6f}')
        if snapshot_prefix is not None and epoch+1 != max_epochs:
            save_model(model, snapshot_prefix, snapshot_number=epoch)

    return model

def save_model(model, filename, snapshot_number=-1):
    model = get_copycat_model(model)
    if snapshot_number != -1:
        filename = f'{filename.split(".pth")[0]}-snapshot_{snapshot_number:03d}.pth'
    else:
        filename = f'{filename.split(".pth")[0]}.pth'
    check_file_and_rename(filename)
    torch.save(model.state_dict(), filename)

def test(model, dataset, db_name='test', batch_size=32, metric='f1_score'):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model = get_copycat_model(model)
    model  = model.to(device)
    db     = getattr(dataset, db_name)
    loader = getattr(dataset, f'{db_name}_loader')
    loader = loader(batch_size=batch_size)

    y_true = np.zeros([len(db)], dtype=np.int)
    y_pred = np.zeros([len(db)], dtype=np.int)
    ans_pos = 0
    with torch.no_grad():
        model.eval()
        with tqdm(loader) as tqdm_test:
            tqdm_test.set_description('[ Testing ]')
            for data in tqdm_test:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                predictions = outputs.data.max(dim=1)[1]
                #saving true labels and predictions
                begin   = ans_pos
                end     = ans_pos+batch_size
                ans_pos = end
                y_true[begin:end] = labels.detach().cpu()
                y_pred[begin:end] = predictions.detach().cpu()

    if type(metric) == list or type(metric) == tuple:
        ret = [compute_metrics(y_true, y_pred, metric=met) for met in metric]
    else:
        ret = compute_metrics(y_true, y_pred, metric=metric)
    return ret

def compute_metrics(y_true, y_pred, metric='f1_score', digits=6):
    assert metric == 'f1_score' or metric == 'accuracy' or \
           metric == 'plain' or metric == 'report', 'The "metric" must be "f1_score", "accuracy", "plain", or "report"'
    if metric == 'f1_score':
        micro_avg = f1_score(y_true, y_pred, average='micro')
        macro_avg = f1_score(y_true, y_pred, average='macro')
        return micro_avg, macro_avg
    elif metric == 'accuracy':
        micro_avg = accuracy_score(y_true, y_pred)
        macro_avg = balanced_accuracy_score(y_true, y_pred)
        return micro_avg, macro_avg
    elif metric == 'plain':
        return y_true, y_pred
    elif metric == 'report':
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred, digits=digits)
        bac = balanced_accuracy_score(y_true, y_pred)
        f1_macro_avg = f1_score(y_true, y_pred, average='macro')
        max_cat  = int(y_true.max())+1
        col_size = len(str(cm.max()))+1
        cat_size = len(str(max_cat))+2
        ret  = 'Confusion Matrix:\n'
        ret += f'{" ":{cat_size+2}s}'
        ret += ''.join([ f'{x:{col_size}d}' for x in range(max_cat) ]) + '\n'
        for line in range(max_cat):
            ret += f'{line:{cat_size}d}: '
            ret += ''.join([f'{x:{col_size}d}' for x in cm[line,:]]) + '\n'
        ret += '-'*(cat_size+col_size*max_cat+2) + '\n'
        ret += cr
        ret += '-'*len(cr.split('\n')[-2]) + '\n'
        ret += f'Accuracy    Macro Average: {bac:.{digits}f}\n'
        ret += f'Accuracy F1-Macro Average: {f1_macro_avg:.{digits}f}'
        return ret

def label(model, dataset, shuffle=False, db_name='npd', batch_size=32, hard_labels=True):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model = get_copycat_model(model)
    model  = model.to(device)
    db     = getattr(dataset, db_name)
    loader = getattr(dataset, f'{db_name}_loader')
    loader = loader(shuffle=shuffle, batch_size=batch_size)
    db.ret_fn()

    filenames   = np.empty((0), dtype=np.str)
    predictions = np.empty((0), dtype=np.int if hard_labels else tuple)
    with torch.no_grad():
        model.eval()
        with tqdm(loader) as tqdm_loader:
            tqdm_loader.set_description(f'[ Labeling dataset "{db_name}" ]')
            for images, _, batch_filenames in tqdm_loader:
                images = images.to(device)
                outputs = model(images).detach().cpu().numpy()
                filenames = np.append(filenames, list(batch_filenames), axis=0)
                if hard_labels:
                    outputs = outputs.argmax(axis=1)
                else:
                    outputs = outputs.view([('', outputs.dtype)]*outputs.shape[1]).reshape(outputs.shape[0])
                predictions = np.append(predictions, outputs, axis=0)

    return np.rec.fromarrays( [filenames, predictions], names=('filenames', 'predictions') )

def label_image(model, image, hard_labels=True):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model  = get_copycat_model(model)
    model  = model.to(device)
    image  = image.to(device)
    if len(image.shape) == 3 or len(image.shape) == 1:
        image = image.unsqueeze(dim=0)
    with torch.no_grad():
        model.eval()
        output = model(image).detach().cpu().squeeze().numpy()
        return output.argmax() if hard_labels else output
