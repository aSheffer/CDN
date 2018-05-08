# Notebooks

This folder contains all the project's notebooks with which you can examine the results and test everything yourself (Some graphs and tables might be more readable after downloading the notebooks to your on computer). Note that all the tests outputs can be found in ```root/notebooks/logs/``` (where ```root``` is the project root directory) and the code of the models themselves can be found in ```root/models/``` .

### Notebooks Overview


- ```SG.ipynb``` shows the results of SG without BN (Batch Normalization) nor CDN (Cross Domain Normalization). We test the model with different hyperparameters and different mechanisms (bidirectional RNN and attention and both). 
  
- Next you'll find in ```SGBN.ipynb``` the results for SG with BN over the image domain, language domain and both. In these experiments the number of the LSTM hidden units is set to 50, 100 and 150.
 
- In ```CDS_Analysis.ipynb``` we did the same experiments as in ```SGBN.ipynb``` while setting the LSTM cell to 200. However, this time we've collected the features ranges, variances and the update rates of the different domains. These results are further discussed in the paper.

- In ```SBN_Analysis.ipynb``` we've analysed the performances, statistics and update rates of SG with SBN (Scaled Batch Nornalization).
  
- We've then tested SG with CDN in ```CDN_Analysis.ipynb```. We've also tried CDN with attention, bidirectional RNN and both. Again, we've gathered and analysed the domains statistics and update rates for each experiment.

- For ablation, in ```NoSpat.ipynb``` we've tested the effect of removing the spatial features. In ```imagesOnly.ipynb``` we've removed the textual queries altogether, leaving only the images. This helps us understand the amount of information we can extract from the images alone and the model's ability to overfit the images.

- Finally, we've added two baselines which you can find in ```Rand.ipynb``` and ```CBoWG.ipynb```.

- In ```demo.ipynb``` you'll find our demo where you can run our model (CDND) with different images. ```root/demo_images/``` contains some images with which you can run the demo. Note that the demo uses Mask-RCNN (with Tensorflow), thus, you'll need to have the proper installation (see the README file in the project root).
