# Flex-U-Net
 
Welcome to the the official github repository for Flex-U-Net. Here you can download the essential codes and files for running and using Flex-U-Net as a new architecture for semantic segmentation of any kind of objective targets especially the biomedical ones.
    
## Running Codes
> All the codes have been tested on Windows 10 using Anaconda.

> For running the code, you should first install all the libraries and frameworks mentioned in the requirements.txt file. For this goal, you just need to run the following code in your command window for your active environment:
>
>     pip install --user -r requirements.txt

> The other codes in this repository are:
> 1. Train_Test_Eval_Program.py
> 2. FlexUNET_Model.py

* Note1: To deploy the Flex-U-Net architecture for training the model on your own dataset, you must apply some changes to the content of the "# Loading Data for train, validation and test procedures" field in **Train_Test_Eval_Program.py** code.
* Note2: The definition of specific hyperparameters related to Flex-U-Net such as NN, KK, dd, and the backbone can be defined under "# Hyperparameter adjustment" section in **Train_Test_Eval_Program.py** code.
* Note3: The training related parameters can be defined under "# Adjustment of training Parameters" section of **Train_Test_Eval_Program.py** code.
* Note4: In the current code format, for simplicity the patch size is the same as the size of original image to keep the aspect ratio and the due to the hardware limitations.


> Cheers,

> Ashkan
>


