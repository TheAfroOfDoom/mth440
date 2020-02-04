# mth440
Projects done during MTH 440 in Fall 2019 at UMassD with the ultimate goal of trying to make the process in which we train networks to detect gravitational waves more efficient.

### waves.py:
Generates 60000 training images and 6000 test images of sine, gaussian, and quadratic waves.  Then, uses tensorflow to train a network with the data to classify between the different types of waves.

![](https://i.imgur.com/RE6yFCS.png)  
![](https://i.imgur.com/7DgNrVi.png)  

### rademacher.py:
My first attempt to calculate the Rademacher value of a set of functions.  This version is extremely slow due to not utilizing _numpy_ properly, typically taking days to calculate R for even simple functions.

![](https://i.imgur.com/3bCU0JG.png)  

### rademacher_2.py:
This second version to calculate the Rademacher value of a set of functions runs at a reasonable speed, calculating R for three simple functions in only a few minutes.

![](https://i.imgur.com/8C5rkTH.png)  
![](https://i.imgur.com/YWjuiQa.png)  
