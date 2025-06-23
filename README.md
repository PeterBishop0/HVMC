# Dataset (HVMC) for Harmful Video Multi-label Classification


## Dataset Overview

### Dataset Introduction
To address the challenges of ambiguous labeling system definition and scarcity of high-quality training data for multi-label classification of harmful videos, we construct **a first large-scale multi-label classification dataset of harmful videos, HVMC**. 

HVMC consists of short videos of 3-20s, which contain both harmful and normal categories, where the harmful categories include gore, violence, nudity and pornography, political sensitivity, abusive, money risk, and smoking and drug. HVMC selected these labels for the **Harmful Video category** because these videos pose a significant content security risk to major video platforms.  There are a short description of each harmful category belowing:

**(1) Gory:** This category contains visual elements such as large amounts of blood spatter, broken limbs, exposed internal organs, close-ups of wounds, bloody scenes, and severe physical trauma caused by violence.

**(2) Violence:** This category includes acts of violent conflict that directly or indirectly cause bodily injury or death, such as beatings, killing, torture, abuse, and assault with weapons.

**(3) Nudity and Pornography:** This category includes direct sexual acts, nudity of key body parts, close-ups of sexual organs, and other explicit images designed to arouse sexual excitement.

**(4) Politically Sensitive:** This category includes insulting images or inappropriate remarks directed at the People's Republic of China, such as stigmatization of political figures, stigmatization of the party and the country, and distortion of political positions.

**(5) Abusive:** This category contains directly insulting remarks or abusive remarks about others.

**(6) Money Risk:** This category contains advertisements that induce betting, gambling, swiping, and credit.

**(7) Smoking and drug:** this category contains images of smoking or drug use.

**(8) Normal:** This category is for scenes that do not contain any of the above offending video categories.

Here shows the examples of the 8 categories in HVMC:

![HVMC Harmful Video vs. Normal Video Example Chart](https://github.com/PeterBishop0/HVMC/blob/main/images/hvmc_example.png)

You can view the raw frame rate of the video through VideoReader's get_avg_fps() method:

```angular2html
from decord import VideoReader

vr = VideoReader(video_path)
default_fps = vr.get_avg_fps()
print("raw frame rate of video (fps):", default_fps)
```

**For more detailed information on the HVMC dataset, see our paper.**

### Download
If you would like to access the HVMC dataset, please fill out this [google form](https://docs.google.com/forms/d/1nYTahtgAUpe2gsl00TY8rMePY8pQ4XHiLJJEOyVZX3A/edit?usp=forms_home&ths=true). The download link will be sent to you once the form is accepted (in 72 hours). If you have any questions, please send email to [jiawei_ge@seu.edu.com].


## License and Citation

The HVMC database is released only for academic research. Researchers from educational institute are allowed to use this database freely for noncommercial purpose.

If you use this dataset, please cite the following paper:
