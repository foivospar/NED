## Neural Emotion Director (NED) - Official Pytorch Implementation (CVPR 2022 Oral)

<img src="imgs/teaser.gif" width="100%" height="100%"/>
Example video of facial emotion manipulation while retaining the original mouth motion, i.e. speech. We show examples of 3 basic emotions.

<br><br>
This repository contains the source code for our paper:
> **Neural Emotion Director: Speech-preserving semantic control of facial expressions in “in-the-wild” videos**<br>
> Foivos Paraperas Papantoniou, Panagiotis P. Filntisis, Petros Maragos, Anastasios Roussos<br>

> Project site:
https://foivospar.github.io/NED/<br>

> **Abstract:** *In this paper, we introduce a novel deep learning method for photo-realistic manipulation of the emotional state of actors in ``in-the-wild'' videos. The proposed method is based on a parametric 3D face representation of the actor in the input scene that offers a reliable disentanglement of the facial identity from the head pose and facial expressions. It then uses a novel deep domain translation framework that alters the facial expressions in a consistent and plausible manner, taking into account their dynamics. Finally, the altered facial expressions are used to photo-realistically manipulate the facial region in the input scene based on an especially-designed neural face renderer. To the best of our knowledge, our method is the first to be capable of controlling the actor’s facial expressions by even using as a sole input the semantic labels of the manipulated emotions, while at the same time preserving the speech-related lip movements. We conduct extensive qualitative and quantitative evaluations and comparisons, which demonstrate the effectiveness of our approach and the especially promising results that we obtain. Our method opens a plethora of new possibilities for useful applications of neural rendering technologies, ranging from movie post-production and video games to photo-realistic affective avatars.*

## Getting Started
Clone the repo:
  ```bash
  git clone https://github.com/foivospar/NED
  cd NED
  ```  

### Requirements
Create a conda environment, using the provided ```environment.yml``` file.
```bash
conda env create -f environment.yml
```
Activate the environment.
```bash
conda activate NED
```

### Files
1. Follow the instructions in [DECA](https://github.com/YadiraF/DECA) (under the *Prepare data* section) to acquire the 3 files ('generic_model.pkl', 'deca_model.tar', 'FLAME_albedo_from_BFM.npz') and place them under "./DECA/data".
2. Fill out the [form](https://docs.google.com/forms/d/e/1FAIpQLScyyNWoFvyaxxfyaPLnCIAxXgdxLEMwR9Sayjh3JpWseuYlOA/viewform) to get access to the [FSGAN](https://github.com/YuvalNirkin/fsgan)'s pretrained models. Then download 'lfw_figaro_unet_256_2_0_segmentation_v1.pth' (from the "v1" folder) and place it under "./preprocessing/segmentation".

## Video preprocessing
To train or test the method on a specific subject, first create a folder for this subject and place the video(s) of this subject into a **"videos"** subfolder. To acquire the training/test videos used in our experiments, please [contact us](mailto:phoebus1998p@gmail.com).

For example, for testing the method on Tarantino's clip, a structure similar to the following must be created:
```
Tarantino ----- videos ----- Tarantino_t.mp4
```
Under the above structure, there are 3 options for the video(s) placed in the "videos" subfolder:
1. Use it as test footage for this actor and apply our method for manipulating his/her emotion.
2. Use this footage to train a neural face renderer on the actor (e.g. use the training video one of our 6 YouTube actors, or a footage of similar duration for a new identity).
3. Use it only as reference clip for transferring the expressive style of the actor to another subject.

To preprocess the video (face detection, segmentation, landmark detection, 3D reconstruction, alignment) run:
```bash
./preprocess.sh <celeb_path> <mode>
```
- ```<celeb_path>``` is the path to the folder used for this actor.
- ```<mode>``` is one of ```{train, test, reference}``` for each of the above cases respectively.

After successfull execution, the following structure will be created:

```
<celeb_path> ----- videos -----video.mp4 (e.g. "Tarantino_t.mp4")
                   |        |
                   |        ---video.txt (e.g. "Tarantino_t.txt", stores the per-frame bounding boxes, created only if mode=test)
                   |
                   --- images (cropped and resized images)
                   |
                   --- full_frames (original frames of the video, created only if mode=test or mode=reference)
                   |
                   --- eye_landmarks (landmarks for the left and right eyes, created only if mode=train or mode=test)
                   |
                   --- eye_landmarks_aligned (same as above, but aligned)
                   |
                   --- align_transforms (similarity transformation matrices, created only if mode=train or mode=test)
                   |
                   --- faces (segmented images of the face, created only if mode=train or mode=test)
                   |
                   --- faces_aligned (same as above, but aligned)
                   |
                   --- masks (binary face masks, created only if mode=train or mode=test)
                   |
                   --- masks_aligned (same as above, but aligned)
                   |
                   --- DECA (3D face model parameters)
                   |
                   --- nmfcs (NMFC images, created only if mode=train or mode=test)
                   |
                   --- nmfcs_aligned (same as above, but aligned)
                   |
                   --- shapes (detailed shape images, created only if mode=train or mode=test)
                   |
                   --- shapes_aligned (same as above, but aligned)
```
## 1.Manipulate the emotion on a test video
Download our pretrained manipulator from [here](https://drive.google.com/drive/folders/1fLAsB2msBcLnRJWlixXt-hJ8FeX3Az6T?usp=sharing) and unzip the checkpoint. We currently provide only the test scripts for the manipulator.

Also, preprocess the test video for one of our target YouTube actors or use a new actor (requires training a new neural face renderer).

For our YouTube actors, we provide pretrained renderer models [here](https://drive.google.com/drive/folders/1vBVeiBvVP_fZ5jPSv7yd7OsdiI22Mwnd?usp=sharing). Download the .zip file for the desired actor and unzip it.

Then, assuming that preprocessing (in **test** mode) has been performed for the selected test video (see above), you can manipulate the expressions of the celebrity in this video by one of the following 2 ways:

##### 1.Label-driven manipulation
Select one of the 7 basic emotions (happy, angry, surprised, neutral, fear, sad, disgusted) and run :
```bash
python manipulator/test.py --celeb <celeb_path> --checkpoints_dir ./manipulator_checkpoints --trg_emotions <emotions> --exp_name <exp_name>
```
- ```<celeb_path>``` is the path to the folder used for this actor's test footage (e.g. "./Tarantino").
- ```<emotions>``` is one or more of the 7 emotions. If one emotion is given, e.g. ```--trg_emotions happy```, all the video will be converted to happy, whereas for 2 or more emotions, such as ```--trg_emotions happy angry``` the first half of the video will be happy, the second half angry and so on.
- ```<exp_name>``` is the name of the sub-folder that will be created under the <celeb_path> for storing the results.

##### 2.Reference-driven manipulation
In this case, the reference video should first be preprocessed (see above) in **reference** mode. Then run:
```bash
python manipulator/test.py --celeb <celeb_path> --checkpoints_dir ./manipulator_checkpoints --ref_dirs <ref_dirs> --exp_name <exp_name>
```
- ```<celeb_path>``` is the path to the folder used for this actor's test footage (e.g. "./Tarantino").
- ```<ref_dirs>``` is one or more reference videos. In particular, the path to the "DECA" sublfolder has to be given. As with labels, more than one paths can be given, in which case the video will be transformed sequentially according to those reference styles.
- ```<exp_name>``` is the name of the sub-folder that will be created under the <celeb_path> for storing the results.




Then, run:
```bash
./postprocess.sh <celeb_path> <exp_name> <checkpoints_dir>
```
- ```<celeb_path>``` is the path to the test folder used for this actor.
- ```<exp_name>``` is the name you have given to the experiment in the previous step.
- ```<checkpoints_dir>``` is the path to the pretrained renderer for this actor (e.g. "./checkpoints_tarantino" for Tarantino).

This step performs neural rendering, un-alignment and blending of the modified faces. Finally, you should see the ```full_frames``` sub-folder into ```<celeb_path>/<exp_name>```. This contains the full frames of the video with the altered emotion. To convert them to video, run:
```bash
python postprocessing/images2video.py --imgs_path <full_frames_path> --out_path <out_path> --audio <original_video_path>
```
- ```<full_frames_path>``` is the path to the full frames (e.g. "./Tarantino/happy/full_frames").
- ```<out_path>``` is the path for saving the video (e.g. "./Tarantino_happy.mp4").
- ```<original_video_path>``` is the path to the original video (e.g. "./Tarantino/videos/tarantino_t.mp4"). This argment is optional and is used to add the original audio to the generated video.


## 2.Train a neural face renderer for a new celebrity
Download our pretrained meta-renderer ("checkpoints_meta-renderer.zip") from the link above and unzip the checkpoints.

Assuming that the training video of the new actor has been preprocessed (in **train** mode) as described above, you can then finetune our meta-renderer on this actor by running:
```bash
python renderer/train.py --celeb <celeb_path> --checkpoints_dir <checkpoints_dir> --load_pretrain <pretrain_checkpoints> --which_epoch 15
```
- ```<celeb_path>``` is the path to the train folder used for the new actor.
- ```<checkpoints_dir>``` is the new path where the checkpoints will be saved.
- ```<load_pretrain>``` is the path with the checkpoints of the pretrained meta-renderer (e.g. "./checkpoints_meta-renderer")

## 3.Preprocess a reference video
If you want to use a reference clip (e.g. from a movie) of another actor to transfer his/her speaking style to your test actor, simply preprocess the reference actor's clip as described above (mode=*reference*) and follow the instructions on **Reference-driven manipulation**.

## Citation


If you find this work useful for your research, please cite our paper. 

```
@inproceedings{paraperas2022ned,
         title={Neural Emotion Director: Speech-preserving semantic control of facial expressions in "in-the-wild" videos}, 
         author={Paraperas Papantoniou, Foivos and Filntisis, Panagiotis P. and Maragos, Petros and Roussos, Anastasios},
         booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
         year={2022}
}
```

## Acknowledgements
We would like to thank the following great repositories that our code borrows from:
- [Head2Head](https://github.com/michaildoukas/head2head)
- [StarGAN v2](https://github.com/clovaai/stargan-v2)
- [DECA](https://github.com/YadiraF/DECA)
- [FSGAN](https://github.com/YuvalNirkin/fsgan)
