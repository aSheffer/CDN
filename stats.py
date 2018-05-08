import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def stats(test, train, epochs=100, title=None, params=[50, 100, 150, 200], 
          test_Glabels = ['test accuracy', 'test P@1', 'test loss'],
          train_Glabels = ['train accuracy', 'train P@1', 'train loss'],
          path=None):
    '''
    Plot graphs and print the results.
    By default this method aims to plot the loss, accuracy and P@1 on the 
    test and train sets. 
    
    Params:
        test: list. 
              Each item is a tuple, [test accuracy, test P@1, test loss]
        
        train: list. 
               Each item is a tuple, [train accuracy, train P@1, train loss]
               
        params: The hyper-parameters to iterate over, default to number of rnn's hidden units.
        
        path: Where to save the graphs image. If none, do not save. 
        
        test_Glabels/train_Glabels: Plots labels for test/train set results. The method 
                                    will print the best results for each label (accuracy,
                                    P@1 and loss as default)
    '''
    
    epochs = range(epochs)
    test_res = np.array(test)
    train_res = np.array(train)
    figs = []
    
    
    for j, param in enumerate(params):
        f, P = plt.subplots(1, 3, figsize=(12,4))
       
        for i in range(len(train_Glabels)):
            P[i].plot(epochs, test_res[j][:,i])
            P[i].plot(epochs, train_res[j][:,i])
            P[i].legend([test_Glabels[i], train_Glabels[i]], loc=0)
            if title is not None:
                P[i].set_title('%s'%(title))

            # metric = 'accuracy' / 'P@1' / 'loss'
            metric = ''.join(train_Glabels[i][len('train')+1:])
            if metric=='loss':
                print('Train loss %s:%.3f'%(metric, min(train_res[j][:,i])))
                print('Test loss %s:%.3f'%(metric, min(test_res[j][:,i])))
            else:
                print('Train %s:%.3f'%(metric, max(train_res[j][:,i])))
                print('Test %s:%.3f'%(metric, max(test_res[j][:,i])))
            P[i].plot()
            
        plt.show()
        
        if path is not None:
            f.savefig(path+str(params[j])+'_plot.png')
        figs.append(f)
        print('-'*100,'\n')
    return figs