# StyleGAN  
<br> 

[This](https://github.com/NVlabs/stylegan) repository contains the official TensorFlow implementation of the following paper:  
  
> **A Style-Based Generator Architecture for Generative Adversarial Networks**<br>
> Tero Karras (NVIDIA), Samuli Laine (NVIDIA), Timo Aila (NVIDIA)<br>
> http://stylegan.xyz/paper
>
> **Abstract:** *We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis. The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation. To quantify interpolation quality and disentanglement, we propose two new, automated methods that are applicable to any generator architecture. Finally, we introduce a new, highly varied and high-quality dataset of human faces.*  
  
<br>  

[This](https://github.com/Puzer/stylegan) repository contains the encoder for Official TensorFlow Implementation. It's used to converts real images to latent space that can be feed into StyleGAN.

<br>

### Generating latent representation 
```
python encode_images.py aligned_images/ generated_images/ latent_representations/
```
[encode_images.py](https://github.com/Puzer/stylegan/blob/master/encode_images.py) takes four useful parameters:  
>'--image_size', default=256, help='Size of images for perceptual model', type=int  
'--lr', default=1., help='Learning rate for perceptual model', type=float  
'--iterations', default=1000, help='Number of optimization steps for each batch', type=int  
'--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool  

<br> 

#### Hyperparameter tuning
parameter - loss  
--default - 0.25  
--randomize_noise True - 0.64  
--lr 0.5 - 0.24  
--lr 1.5 - 0.22  
--image_size 1024 - 0.23  
--iterations 3000 - 0.16  
--iterations 6000 - 0.14  

Increasing iterations is the only way to decrease loss.  
![](/images/lossVsIters.jpg)

Examples of how loss decreases when iterations increase  
![](/images/lossVsItersExample.jpg)

<br> 

### Kaur minus Leo
```
type(kaur), kaur.shape
>>> (numpy.ndarray, (18, 512))
display(kaur), display(leo), display(leo - kaur)
```
![](/images/kaur.png)![](/images/leo.png)![](/images/diff.png)

<br> 

### Kaur to Leo
![](/images/kaurLeo.gif)

<br> 

### Style mixing
Result when the 8 input images were trained with defualt settings:  
[58MB image in Google Drive](https://drive.google.com/file/d/17H_Faxs_yvidOIhofHeCvFTMC6zBTVr-/view?usp=sharing)

Result when the images were trained 4000 iterations:  
[59MB image in Google Drive](https://drive.google.com/file/d/1JnRg-R2IltIjujDvuXgJsr7DCVd_-E_Q/view?usp=sharing)

<br> 

### Style transformation
In general it's possible to find directions of almost any face attributes: position, hair style or color etc.

#### Age
![](/images/age.jpg)

#### Gender
![](/images/gender.jpg)

#### Smile
![](/images/smile.jpg)
![](/images/horror.jpg)
