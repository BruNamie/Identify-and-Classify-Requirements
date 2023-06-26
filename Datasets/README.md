# Datasets
This folder contains training and testing datasets for each 10-fold Cross-Validation iteration. 


## Training dataset examples
The "Dataset_train_eBay.xlsx" contains reviews from 7 mobile applications for training when the test dataset uses eBay reviews ("Dataset_test_eBay.xlsx").

The "Dataset_train_Evernote.xlsx" contains reviews from 7 mobile applications for training when the test dataset uses Evernote reviews ("Dataset_test_Evernote.xlsx").

## Test datasets
The folder "Test dataset" contains datasets for testing the model for each iteration.
There are three types of datasets: 
* Dataset_test_[app]: reviews from the specific app
* Dataset_test_extracted_[app]: reviews from the specific app that have requirements (RE-BERT model output)
* Dataset_test_others_[app]: reviews from the specific app that don't have requirements (RE-BERT model output)
