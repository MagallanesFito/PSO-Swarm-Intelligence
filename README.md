# PSO-Swarm-Intelligence
A modified version of particle swarm optimization (MPSO) solving a scheduling problem in a reference book.

# The problem

The problem is given in the book __Programming Collective Intelligence__ by Toby Segaran, in chapter 5: Optimization. The problem is about a group travel, your task is try the best combination which gives the minimum cost (travel cost and wait time). A list of optimization methods is proposed, including Genetic Algorithm, Simulated Annealing and Hill Climbing. In this repo the Binary Discrete Particle Swarm Optimization is proposed, and compared with Genetic Algorithms. Note: __Family members names were changed for some friend names just for fun, but the problem is essentialy the same__ :).

# The paper

This was a scholar project, the report is in __report.pdf__ (In spanish). 

# The project

The code folder contains `main.py`, this is the file where all the magic happens, the remaining files are for debugging and testing. Main file has psooptimize() fucntion, this is a naive version of MBPSO, didn't get good results. The one that makes the job is `kbpsooptimize()` which is adapted for Binary PSO. The code section which calls the function looks messy because it is coded for making a grid parameter search and print results, graphs and stuff like that,  you can comment this part and just call the function. The file `schedule.txt` contains the dataset provided by the book. 
