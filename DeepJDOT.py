# -*- coding: utf-8 -*-
"""
Created on Saturday 28.05.2021

Paper and original tensorflow implementation: damodara


@author: marc seibel
Deepjdot  - class file

This is a translation from the original tensorflow implementation into pytorch.
"""
import torch
import numpy as np
import ot
import tqdm

class Deepjdot(object):
    def __init__(self, model, batch_size, n_class, optim=None, allign_loss=1.0, tar_cl_loss=1.0, 
                 sloss=0.0,tloss=1.0,int_lr=0.01, ot_method='emd',
                 jdot_alpha=0.01, lr_decay=True, verbose=1):
        
        self.net        = model   # target model
        self.batch_size = batch_size
        self.n_class    = n_class
        if optim is not None:
            raise ValueError("A custom optimizer is not implemented yet")
            self.optimizer = optim
        # initialize the gamma (coupling in OT) with zeros
        self.gamma = torch.zeros(size=(self.batch_size, self.batch_size))
        # whether to minimize with classification loss
        
        self.train_cl   = torch.tensor(tar_cl_loss) # translated from K.variable
        # whether to minimize with the allignment loss 
        self.train_algn = torch.tensor(allign_loss)
        self.sloss      = torch.tensor(sloss) # weight for source classification
        self.tloss      = torch.tensor(tloss) # weight for target classification
        
        self.verbose = verbose
        self.int_lr  = int_lr  # initial learning rate
        self.lr_decay= lr_decay
        #
        self.ot_method = ot_method
        self.jdot_alpha=jdot_alpha  # weight for the alpha term
        
        
        # target classification cross ent loss and source cross entropy
        def classifier_cat_loss(source_ypred, ypred_t, ys):
            '''
            classifier loss based on categorical cross entropy in the target domain
            y_true:  
            y_pred: pytorch tensor which has gradients
            
            0:batch_size - is source samples
            batch_size:end - is target samples
            self.gamma - is the optimal transport plan
            '''   
            # pytorch has the mean-inbuilt, 
            source_loss = torch.nn.functional.cross_entropy(source_ypred,
                                                            torch.argmax(ys,dim=1)) 
            
            
            # categorical cross entropy loss
            ypred_t = torch.log(ypred_t)
            # loss calculation based on double sum (sum_ij (ys^i, ypred_t^j))
            loss = -torch.matmul(ys, torch.transpose(ypred_t,1,0))
            # returns source loss + target loss
            
            # todo: check function of tloss train_cl, and sloss
            return self.train_cl*(self.tloss*torch.sum(self.gamma * loss) + self.sloss*source_loss)
        self.classifier_cat_loss = classifier_cat_loss
        
        # L2 distance
        def L2_dist(x,y):
            '''
            compute the squared L2 distance between two matrics
            '''
            distx = torch.reshape(torch.sum(torch.square(x),1), (-1,1))
            disty = torch.reshape(torch.sum(torch.square(y),1), (1,-1))
            dist = distx + disty
            dist -= 2.0*torch.matmul(x, torch.transpose(y,0,1))  
            return dist
            
       # feature allignment loss
        def align_loss(g_source, g_target):
            '''
            source and target alignment loss in the intermediate layers of the target model
            allignment is performed in the target model (both source and target features are from target model)
            y-pred - is the value of intermediate layers in the target model
            1:batch_size - is source samples
            batch_size:end - is target samples            
            '''
            # source domain features            
            #gs = y_pred[:batch_size,:] # this should not work????
            # target domain features
            #gt = y_pred[batch_size:,:]
            gdist = L2_dist(g_source,g_target)  
            return self.jdot_alpha * torch.sum(self.gamma * (gdist))
        self.align_loss= align_loss
        
        def feature_extraction(model, data, out_layer_num=-2):
            '''
            Chop simple sequential model from layer 0 to out_layer_num.
            
            # https://discuss.pytorch.org/t/is-it-possible-to-slice-a-model-at-an-arbitrary-layer/53766
            comment: This method has no internal usage.
            
            
            
            extract the features from the pre-trained model
            inp_layer_num - input layer
            out_layer_num -- from which layer to extract the features
            '''

            intermediate_layer_model = model[:out_layer_num]
            intermediate_output = intermediate_layer_model(data)
            return intermediate_output
        self.feature_extraction = feature_extraction
 

 
    def fit(self, source_traindata, ys_label, target_traindata, target_label = None,
            n_iter=5000, cal_bal=True, sample_size=None):
        '''
        source_traindata - source domain training data
        ys_label - source data true labels
        target_traindata - target domain training data
        cal_bal - True: source domain samples are equally represented from
                        all the classes in the mini-batch (that is, n samples from each class)
                - False: source domain samples are randomly sampled
        target_label - is not None  : compute the target accuracy over the iterations
        '''
      
        ns = source_traindata.shape[0]
        nt = target_traindata.shape[0]
        method  = self.ot_method # for optimal transport
        alpha   = self.jdot_alpha
        t_acc  = []
        g_metric ='deep' # to allign in intermediate layers, when g_metric='original', the
         # alignment loss is performed wrt original input features  (StochJDOT)
        
        # function to sample n samples from each class
        def mini_batch_class_balanced(label, sample_size=20, shuffle=False):
            ''' sample the mini-batch with class balanced
            '''
            label = np.argmax(label, axis=1)
            if shuffle:
                rindex = np.random.permutation(len(label))
                label = label[rindex]

            n_class = len(np.unique(label))
            index = []
            for i in range(n_class):
                s_index = np.nonzero(label == i)
                s_ind = np.random.permutation(s_index[0])
                index = np.append(index, s_ind[0:sample_size])
                #          print(index)
            index = np.array(index, dtype=int)
            return index
            
         # target model compliation and optimizer
        optimizer = torch.optim.SGD(self.net.parameters(), self.int_lr)
        
        
        cat_losses   = []
        align_losses = []
        with tqdm.tqdm(range(n_iter), unit='batch') as tepoch:
            for i in tepoch:
                
                if self.lr_decay and i > 0 and i%5000 ==0:
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr']*0.1
    
                # source domain mini-batch indexes
                if cal_bal:
                    s_ind = mini_batch_class_balanced(ys_label, sample_size=sample_size)
                    self.sbatch_size = len(s_ind)
                else:
                    s_ind = np.random.choice(ns, self.batch_size)
                    self.sbatch_size = self.batch_size
                    # target domain mini-batch indexes
                t_ind = np.random.choice(nt, self.batch_size)
    
                # source and target domain mini-batch samples 
                xs_batch = torch.tensor(source_traindata[s_ind]).type(torch.float)
                ys       = torch.tensor(ys_label[s_ind])
                xt_batch = torch.tensor(target_traindata[t_ind]).type(torch.float)
                def to_categorical(y, num_classes):
                    """ 1-hot encodes a tensor """
                    return torch.eye(num_classes, dtype=torch.int8)[y]
                ys_cat = to_categorical(ys,3)
                s = xs_batch.shape
                
                batch = torch.vstack((xs_batch, xt_batch))
                #batch.to(device)
                
                
                self.net.eval() # sets BatchNorm and Dropout in Test mode 
                # concat of source and target samples and prediction
                with torch.no_grad():
                    modelpred = self.net(batch)
                   
                    # modelpred[0] - is softmax prob, and modelpred[1] - is intermediate layer
                    gs_batch = modelpred[1][:self.batch_size, :]
                    gt_batch = modelpred[1][self.batch_size:, :]
                    # softmax prediction of target samples
                    fs_pred  = modelpred[0][:self.batch_size,:]
                    ft_pred  = modelpred[0][self.batch_size:,:]
                    
                    if g_metric=='orginal':
                        # compution distance metric in the image space
                        if len(s) == 3:  # when the input is image, convert into 2D matrix
                            C0 = torch.cdist(xs_batch.reshape(-1, s[1] * s[2]),
                                             xt_batch.reshape(-1, s[1] * s[2]), 
                                             p=2.0)**2
        
                        elif len(s) == 4:
                            C0 = torch.cdist(xs_batch.reshape(-1, s[1] * s[2] * s[3]),
                                             xt_batch.reshape(-1, s[1] * s[2] * s[3]),
                                             p=2.0)**2
                    else:
                        # distance computation between source and target in deep layer
                        C0 = torch.cdist(gs_batch, gt_batch, p=2.0)**2
                    
                    ys_cat = ys_cat.type(torch.float)
                    C1 = torch.cdist(ys_cat, ft_pred, p=2)**2
                    
                    # JDOT ground metric
                    C= alpha*C0+C1
                    
                    # JDOT optimal coupling (gamma)
                    if method == 'emd':
                         gamma=ot.emd(ot.unif(gs_batch.shape[0]),
                                      ot.unif(gt_batch.shape[0]),C)
                    
                    # update the computed gamma                      
                    self.gamma = torch.tensor(gamma)
                

                self.net.train() # Batchnorm and Dropout for train mode
                optimizer.zero_grad()
                # concat of source and target samples and prediction
                modelpred = self.net(batch)
                gs_batch = modelpred[1][:self.batch_size, :]
                gt_batch = modelpred[1][self.batch_size:, :]
                # softmax prediction of target samples
                fs_pred  = modelpred[0][:self.batch_size,:]
                ft_pred  = modelpred[0][self.batch_size:,:]
                

                cat_loss   = self.classifier_cat_loss(fs_pred, ft_pred, ys_cat)
                align_loss = self.align_loss(gs_batch, gt_batch)
                
                loss = cat_loss + align_loss
                #loss = criterion(outputs, batch.y)
                loss.backward()
                optimizer.step()
            
                cat_losses   += [cat_loss.item()]
                align_losses += [align_loss.item()]
                if self.verbose:
                    if i%10==0:
                        cl = np.mean(cat_losses[-10:])
                        al = np.mean(align_losses[-10:])
                        #print('tl_loss ={:f}'.format(cl))
                        #print('fe_loss ={:f}'.format(al))
                        #print('tot_loss={:f}'.format(cl+al))
                        if target_label is not None:
                            tpred = self.net(target_traindata)[0]
                            t_acc.append(torch.mean(target_label==torch.argmax(tpred,1)))
                            print('Target acc\n', t_acc[-1])
                        tepoch.set_postfix(loss=al+cl)
                        
        return [cat_losses,align_losses], t_acc

    def predict(self, data):
        data = torch.tensor(data.astype(np.float32))
        self.net.eval()
        with torch.no_grad():
            ypred = self.net(data)
        return ypred

    def evaluate(self, data, label):
        """
        label as digits (0,1,... , num_classes)
        """
        data = torch.tensor(data).type(torch.float)
        label = torch.tensor(label)
        self.net.eval()
        with torch.no_grad():
            ypred = self.net(data)
        score = torch.mean((label==torch.argmax(ypred[0],1)).type(torch.float))
        return score
