from pathlib import Path
from helper_ftns import compare_folder_images

# compare two folders of images to see if they have similar size distributions
folder1 = Path(r"E:\OneDrive - University of Warwick\RF_Project\PRISM\DATASET\Meso Cells\Epithelioid H266_new")
folder2 = Path(r"E:\OneDrive - University of Warwick\RF_Project\PRISM\DATASET\Meso Cells\Biphasic Cells 33\WT1+CAL")
compare_folder_images(folder1, folder2)

#