for tb and pneumonia dataset:  https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis
from this link download the dataset and delete the covid folder 
the structure for it should look like this and saved in the according names:
tb and pneumonia detection/
│── dataset/
│   ├── train/
│   │   ├── NORMAL/
│   │   ├── PNEUMONIA/
│   │   ├── TUBERCULOSIS/
│   ├── val/
│   │   ├── NORMAL/
│   │   ├── PNEUMONIA/
│   │   ├── TUBERCULOSIS/
│   ├── test/
│   │   ├── NORMAL/
│   │   ├── PNEUMONIA/
│   │   ├── TUBERCULOSIS/
├── train.py      # Training script
├── test.py       # Testing script
├── predict.py    # Prediction script



for the malaraia dataset:    https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria/data
download the dataset and save it as 'cell_images' in the 'malaria detection' folder
and then add the images like p1,p2,...then u1 in the cell_images folder itself-  because i used those images to work , that is how it works now if need to modify note these changes  p-parasitized,u-uninfected


