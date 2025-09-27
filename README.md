# Egocentric Video Understanding with Active Memory

<p align="center">
  <img src="https://github.com/user-attachments/assets/44890bfc-a839-497d-8b49-bede6cb59f96" alt="gud_l">
</p>

This repository contains the code developed for my **Bachelor’s thesis** in Computer Science at the University of Catania. The work focuses on evaluating the performance of [AMEGO-Q5](https://gabrielegoletto.github.io/AMEGO/) query using the [ENIGMA-51](https://iplab.dmi.unict.it/ENIGMA-51/) dataset.

For more details, check the [thesis document](https://github.com/Kespers/Egocentric-Videos-Understanding-with-Active-Memory/blob/23d241f0706d399b3ffdea3af07cd372ba3d74d8/Thesis/document.pdf) or take a look at the [short presentation](https://github.com/Kespers/Egocentric-Videos-Understanding-with-Active-Memory/blob/23d241f0706d399b3ffdea3af07cd372ba3d74d8/presentation/Kevin_Speranza_presentazione-L31.pdf) for a quick summary.

## Relevant Files

- `Experiments/notebook.ipynb`: contains the construction of queries and the evaluation of the results.
- `Experiments/amego_videos.py`: Reads AMEGO HOI tracklets and generates a video displaying the created annotations.  
- `Experiments/amego_frame_visual.py`: Creates a folder structure for each cluster of objects or locations identified by AMEGO.
