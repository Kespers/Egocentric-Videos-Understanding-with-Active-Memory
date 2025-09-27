# Egocentric Video Understanding with Active Memory

![gud](https://github.com/user-attachments/assets/af6d8d35-321c-4464-9d7e-956bb3e2f4eb)

This repository contains code used to evaluate the performance of [AMEGO](https://gabrielegoletto.github.io/AMEGO/)-Q5 queries on the [ENIGMA-51](https://iplab.dmi.unict.it/ENIGMA-51/) dataset.

Included in this repo are the experimental scripts and a [notebook](https://github.com/Kespers/Egocentric-Videos-Understanding-with-Active-Memory/blob/23d241f0706d399b3ffdea3af07cd372ba3d74d8/Experiments/notebook.ipynb) used for generating queries.

For more details, see the [thesis document](https://github.com/Kespers/Egocentric-Videos-Understanding-with-Active-Memory/blob/23d241f0706d399b3ffdea3af07cd372ba3d74d8/Thesis/document.pdf). A short [presentation](https://github.com/Kespers/Egocentric-Videos-Understanding-with-Active-Memory/blob/23d241f0706d399b3ffdea3af07cd372ba3d74d8/presentation/Kevin_Speranza_presentazione-L31.pdf) summarizing the work is also available.

## Relevant Files

- `Experiments/amego_videos.py`: Reads AMEGO HOI tracklets and generates a video displaying the created annotations.  
- `Experiments/amego_frame_visual.py`: Creates a folder structure for each cluster of objects or locations identified by AMEGO.
