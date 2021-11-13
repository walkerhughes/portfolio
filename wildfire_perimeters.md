
## Image Segmentation and Filtering For Estimating Wildfirer Perimeters 

<p>Check it out on my <a href="https://github.com/walkerhughes/wildfire_perimeters">GitHub</a></p>

<script type="text/javascript" async="" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"></script> 

Estimating the true geographic coordinates of a raging wildfire can be difficult to do in real time due to resource constraints of first responders, smoke occlusion in typical RGB images, and various other things. This is an important problem to solve, however, as knowing how a wildfire perimeter progresses over time can give insights to how the fire responds when introduced to different terrains, fuels, and atmospheric conditions. 

In California alone, the U.S. Forest Service spends about $200 million per year to suppress 98% of wildfires and up to $1 billion to suppress the other 2% of fires that escape initial ignition and become large. Being able to detect fires early on and track their real-time growth is not just important for environmental conservation, it's also economically crucial for public service organizations. 

While this is an inherently high dimenional problem, image segmentation provides a simple solution that affords us a high level of interpretability. Here I explore image segmentation and image filtering on Sentinel Satellite images for estimating the perimeter of the Center Creek Trail Fire that burned in Utah in 2020. 

The Sentinel Satellite orbits the earth and takes multispectral images of the planet. These images are high dimensional and contain far more chanels than just Red, Green, and Blue. Let's take a look at the atmospheric channels from a Sentinel Satellite image of this fire. These channels essentially filter through the smoke that would normally occlude an RGB image taken from above, and this allows us to more cllearly view the burned areas. 

Center Creek Trail Fire, burned areas are in orange [Source: Sentinel Satellite] 
<img src="center_creek_burn.jpg" width="750" height="500">   

While we could feasibly use something like a floodfill algorithm at the pixel-color level to determine the perimeter, not all wildfires burn in convex shapes and there are often burned areas not at all connected to the main body of the burn. A lot of images contain unneeded noise (random changes in pixel intensity or brightness) that just aren't neceessary. Image segmentation with additional image filtering is a straightforward way to address this. 

### Pre-Processing with Filters 

We can use Gaussian blurring (gaussian convolution) and Laplace filtering to filter noise from the image and emphasize stark contrasts between pixel intensities. We want to reduce random noise in the image in order to focus on meaningful changes in color intensity (burned areas from not burned). The standard deviation parameter is user-specified.  

Gaussian Convolution Function 

<p><span class="math display">\[ G(x, y) = \frac{1}{2 \pi \sigma^2} e ^ {-\frac{x^2 + y^2}{2 \sigma^2}} \]</span></p> 

Gaussian blurred image
<img src="gaussian_blur_std_8_new.jpg" width="750" height="500">      

We will also use Laplace filtering to emphasize the contrast between pixel intensities. The Laplace filter is a measure of the second spatial derivative on a 2D image and is useful for identifying such areas of rapid pixel intensity changes. This emphasizes fire perimeters clearly in our images. 
 
Laplace Filtering Function

<p><span class="math display">\[ L(x, y) = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2} \]</span></p> 

Below is a single channel of the Laplace filtered image after applying the Gaussian blur. We can clearly see that the burned areas are emphasized in a darker grey compared to the rest of the image. In the original, we had greens, blues, oranges, and various other colors. By isolating specific channels after filtering, we can emphasize the difference between burned areas from raw terrain. 

Laplace filtered image 
<img src="laplacian_filter_new.jpg" width="750" height="500">     

Here are the two filters together - Laaplacian applied to the Gaussian blurred image. White areas are burned, so these differencees are further apparent in this channel of the filtered image.   

<img src="laplace_blur_std_8_new.jpg" width="750" height="500">      

When we complete the final segmentation with a simple numeric cutoff to decide what is burned and what is not, we get the following estimation of the wildfire perimeter. Knowing the geographic coordinates of the edges of the image makes this estimation of the fire perimeter easily mappaple to lattitude and longitude coordinates. 

<img src="final_segmentation_new.jpg" width="750" height="500">      


