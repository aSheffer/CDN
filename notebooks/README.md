# Notebooks

This folder contains all the project's notebooks with which you can examine the results and test everything yourself (Some graphs and tables might be more readable after downloading the notebooks to your on computer). Note that all the tests outputs can be found in ```/notebooks/logs/``` and the code of the models themselves can be found in ```/root/modedels/``` (where ```root``` is the project root directory).

## Notebooks Overview

<ul>
<li> <b>SG.ipynb</b> shows the results of SG without BN (Batch Normalization) nor CDN. We test the model with different hyperparameters and different mechanisms (bidirectional RNN and attention) </li><br>
  
<li> Next you'll find in <b>SGBN.ipynb</b> the results for SG with BN over the image domain, language domain and both. In these experiments the number of the LSTM hidden units is set to 50, 100 and 150.</li><br>  
 
<li> In <b>CDS_Analysis.ipynb</b> we did the same experiments as in  <b>SGBN.ipynb</b> while setting the LSTM cell to 200. However, this time we've collected the features ranges, variances and the update rates of the different domains. These results are further discussed in the paper</li><br>

<li> In <b>SBN_Analysis.ipynb</b> we've analysed the performances, statistics and update rates of SG with SBN (Scaled Batch Nornalization) </li><br>
  
<li> We've then tested CDN (Cross Domain Normalization) in <b>CDN_Analysis.ipynb</b>. We've also tested CDN with attention, bidirectional RNN and both. Again, we've gathered and analysed the domains statistics and update rates for each experiment</li><br>

<li> For ablation, in <b>NoSpat.ipynb</b> we've tested the effect of removing the spatial features. In <b>imagesOnly.ipynb</b> we've removed the textual queries altogether, leaving only the images. This helps us understand the amount of information we can extract from the images alone and the model ability to overfit the images.  <br>

<li>Finally, we've added two baseline which you can find in <b>Rand.ipynb</b> and <b>CBoWG.ipynb</b></li><br>

<li>You can run our model (CDND) in <b>demo.ipynb</b>. Note that the demo uses Mask-RCNN (with Tensorflow), thus, you'll need to have the proper installation (see the README file in the project root)</li>
</ul>
