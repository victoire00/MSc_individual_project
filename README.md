# MSc_individual_project
Imperial College London - MSc 2023 - Biomedical engineering - Individual project on hyperspectral images analysis for neurosurgery

There is a clinical need for a device that can better identify differences between tumour margins, normal brain tissue and functionally active areas during the neurosurgical procedure. The aim of the study is therefore to provide the surgical team with a surgical guidance system. It should be capable of imaging and processing data in real-time and displaying diagnostic data such as brain perfusion (levels of oxygenated/ deoxygenated hemoglobin)

## Aim of the code

Based on methods developed in previous studies, I implemented programs that generate blood oxygenation level maps. I chose to implement a fitting procedure.
- First, I elaborated a model that describes the light transport in biological tissues. For this purpose, a Monte-Carlo simulation has been developed.
- Secondly, I needed a program to fit the spectral data acquired during the surgeries to this model and to generate oxygen saturation
maps.

## Results examples

Some example results can be found in the folder "results_examples"

## How to use the code?

1) Monte-Carlo Simulation (to estimate the mean optical pathlength)
- Set the tissue parameters in the file classes.py (line 49) and in the file functions.py (line 157)
- Run the file montecarlo.py
- The estimated mean optical pathlength is saved as a JSON file in the folder "saved_dpf"

2) Reading one HS image
- in the file main.py edit the path of the image
- Run BLOC 1 of main.py
- The console should tell you the dimensions of the image and how many images have been averaged (1, 2 or 4)
- it also creates a result folder where the results of step 4) will be automatically saved

3) estimating blood oxygenation levels on one pixel
- two different cameras have been used during the study. Whether camera 1 or camera 2 have been used, uncomment the appropriate line in the file functions.py at line 209 and 424
- Run BLOC 2 of main.py
- A fist figure appears, asking the user to click on any pixel.
- then two other figures appear showing the location of the click and a graph of the fited data. The console displays the estimated value for I0, Ct and SO2.

4) generating blood oxygenation level maps
- before doing step 4), verify that step 3) works correctly (the fitting procedure generates satisfying results), if not, you probably selected the wrong camera
- Run BLOC 3 of main.py
- draw a polygon on the image that will be the outline of the region of interest for the maps. each coordinate of the point clicked is saved in a JSON file called polygon in the result folder. You can also chose a tissue segmented by the specialist to be your ROI, by uncommenting the line 78 in main.py
- choose the resolution of the maps line 95 in main.py (ex: if you write 50 it means that the pixels will be averaged by blocs of 50x50 pixels)
- Run bloc 4 of main.py
- The program will first show you the mask used for the map (combination of your region of interest and a mask removing saturated pixels)
- then the maps will be saved in the results folder

5) displaying the maps
- once you have generated the maps using step 4)
- edit the path of your results in the first bloc of maps.py
- run the first bloc to load the results
- the second bloc plots the maps
- the third bloc compares two results. but this is relevant only if the two HS images have been registered using the same field

6) some other interesting results can be displayed by calling the functions in the file display.py
