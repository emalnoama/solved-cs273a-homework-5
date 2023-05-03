Download Link: https://assignmentchef.com/product/solved-cs273a-homework-5
<br>
<h2>1 Clustering</h2>

The code this week provides the three clustering algorithms we discussed: k-means, agglomerative clustering, and

EM for Gaussian mixture models; we will explore the first two here. (These functions are also provided in many

3rd party toolboxes; you are free to use those if you prefer.) In this problem, you’ll do some basic exploration of the clustering techniques.

<ol>

 <li>Load the usual Iris data restricted to the first two features, and ignore the class / target variable. Plot the data and see for yourself how “clustered” you think it looks. Include the plot, and mention how many clusters you think exist (no wrong answer here).</li>

</ol>

<table width="185">

 <tbody>

  <tr>

   <td width="185">ml.plotClassify2D(None,X,z)</td>

  </tr>

 </tbody>

</table>

<ol start="2">

 <li>Run k-means on the data, for <em>k </em>= 2, <em>k </em>= 5, and <em>k </em>= 20. Try a few (at least 5 each) different initializations and check to see whether they find the same solution; if not, pick the one with the best score. For the chosen assignment for each <em>k</em>, include a plot with the data, colored by assignment, and the cluster centers. You can plot the points colored by assignments using, where <em>z </em>are the resulting cluster assignments of the data. You will have to additionally plot the centers yourself. <em>(15 points)</em></li>

</ol>

<table width="271">

 <tbody>

  <tr>

   <td width="165">ml.cluster.agglomerative</td>

   <td width="36">from</td>

   <td width="70">cluster.py</td>

  </tr>

 </tbody>

</table>

<ol start="3">

 <li>Run agglomerative clustering on the data, using <em>single linkage </em>and then again using <em>complete linkage</em>, each with 2, 5, and then 20 clusters (using). Again, plot with color the final assignment of the clusters. (This algorithm has no initialization issues; so you do not have to try multiple initializations.) <em>(20 points)</em></li>

 <li>Describe similarities and differences in the results from the agglomerative clustering and k-means. <em>(5 points)</em></li>

</ol>

<h2>       2       EigenFaces</h2>

In class we discussed that PCA has been applied to faces, and showed some of the results. Here, you’ll explore this representation yourself. First, load the data and display a few faces to make sure you understand the data format:

<table width="624">

 <tbody>

  <tr>

   <td width="624">X = np.genfromtxt(“data/faces.txt”, delimiter=None) # load face dataset plt.figure()# pick a data point i for display img = np.reshape(X[i,:],(24,24)) # convert vectorized data to 24×24 image patches plt.imshow( img.T , cmap=”gray”)                 # display image patch; you may have to squint</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

<ol>

 <li>Subtract the mean of the face images (<em>X</em><sub>0 </sub>= <em>X −µ</em>) to make your data zero-mean. (The mean should be of the same dimension as a face, 576 pixels.) Plot the mean face. <em>(5 points)</em></li>

</ol>

<table width="111">

 <tbody>

  <tr>

   <td width="111">scipy.linalg.svd</td>

  </tr>

 </tbody>

</table>

<ol start="2">

 <li>Useto take the SVD of the data, so that</li>

</ol>

<em>X</em><sub>0 </sub>= <em>U · </em>diag(<em>S</em>)<em>· V<sub>h</sub></em>

<table width="158">

 <tbody>

  <tr>

   <td width="158">W = U.dot( np.diag(S) )</td>

  </tr>

 </tbody>

</table>

Since the number of data is larger than the number of dimensions, there are at most 576 non-zero singular values; you can use full_matrices=False to avoid using a lot of memory. As in the slides, we suggest computingso that <em>X</em><sub>0 </sub><em>≈ W · V<sub>h</sub></em>. Print the shapes of <em>W </em>and <em>V<sub>h</sub></em>. <em>(10 points)</em>

<table width="153">

 <tbody>

  <tr>

   <td width="153">np.mean( (<em>X</em><sub>0                     </sub><em>X</em><sup>ˆ</sup><sub>0</sub><sup>)</sup>**2 )</td>

  </tr>

 </tbody>

</table>

<ol start="3">

 <li>For <em>K </em>= 1 . . . 10, compute the approximation to <em>X</em><sub>0 </sub>given by the first <em>K </em>eigendirections, e.g., <em>X</em>ˆ<sub>0 </sub>= <em>W</em>[:, : <em>K</em>] <em> Vh</em>[: <em>K</em>, :], and use them to compute the mean squared error in the SVD’s approximation,</li>

</ol>

<em>−                         </em>. Plot these MSE values as a function of <em>K</em>. <em>(10 points)</em>

<table width="185">

 <tbody>

  <tr>

   <td width="185">2*np.median(np.abs(W[:,j]))</td>

  </tr>

 </tbody>

</table>

<ol start="4">

 <li>Display the first three principal directions of the data, by computing <em>µ</em>+<em>α </em>V[j,:] and <em>µ</em>–<em>α </em>V[j,:], where <em>α </em>is a scale factor (we suggest, for example,, to get a sense of the scale found in the data). These should be vectors of length 24<sup>2 </sup>= 576, so you can reshape them and view them as “face images” just like the original data. They should be similar to the images in lecture. <em>(10 points)</em></li>

 <li>Choose any two faces and reconstruct them using the first <em>K </em>principal directions, for <em>K </em>= 5, 10, 50, 100. <em>(5 points)</em></li>

 <li>Methods like PCA are often called “latent space” methods, as the coefficients can be interpreted as a new geometric space in which the data are being described. To visualize this, choose a few faces (say 25), and display them as images with the coordinates given by their coefficients on the first two principal components:</li>

</ol>

<table width="591">

 <tbody>

  <tr>

   <td width="591">idx = …                                                                           # pick some data (randomly or otherwise); an array of integer indices<strong>import </strong>mltools.transforms coord,params = ml.transforms.rescale( W[:,0:2] ) # normalize scale of “W” locations plt.figure(); plt.hold(True);          # you may need this for pyplot <strong>for </strong>i <strong>in </strong>idx:# compute where to place image (scaled W values) &amp; size loc = (coord[i,0],coord[i,0]+0.5, coord[i,1],coord[i,1]+0.5)img = np.reshape( X[i,:], (24,24) )     # reshape to square plt.imshow( img.T , cmap=”gray”, extent=loc ) # draw each imageplt.axis( (-2,2,-2,2) )                                                                           # set axis to a reasonable scale</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

8

9

10

11

This can often help you get a “feel” for what the latent representation is capturing. <em>(10 points)</em>

<h2>Statement of Collaboration</h2>

It is <strong>mandatory </strong>to include a <em>Statement of Collaboration </em>in each submission, with respect to the guidelines below. Include the names of everyone involved in the discussions (especially in-person ones), and what was discussed.

All students are required to follow the academic honesty guidelines posted on the course website. For programming assignments, in particular, we encourage the students to organize (perhaps using Piazza) to discuss the task descriptions, requirements, bugs in my code, and the relevant technical content <em>before </em>they start working on it. However, you should not discuss the specific solutions, and, as a guiding principle, you are not allowed to take anything written or drawn away from these discussions (i.e. no photographs of the blackboard, written notes, referring to Piazza, etc.). Especially <em>after </em>you have started working on the assignment, try to restrict the discussion to Piazza as much as possible, so that there is no doubt as to the extent of your collaboration.