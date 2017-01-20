from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

# Load the "gatlin" image data
X = loadtxt('gatlin.csv', delimiter=',')

#================= ADD YOUR CODE HERE ====================================
# Perform SVD decomposition
## TODO: Perform SVD on the X matrix
# Instructions: Perform SVD decomposition of matrix X. Save the 
#               three factors in variables U, S and V
#

U,S,V=linalg.svd(X);

#=========================================================================

# Plot the original image
plt.figure(1)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (rank 480)')
plt.axis('off')
plt.draw()

#================= ADD YOUR CODE HERE ====================================
# Matrix reconstruction using the top k = [10, 20, 50, 100, 200] singular values
## TODO: Create four matrices X10, X20, X50, X100, X200 for each low rank approximation
## using the top k = [10, 20, 50, 100, 200] singlular values 


#X10 reconstruction
U10=U[:,:10];
V10=V[:10,:];
S10=S[:10];
X10=dot(U10,dot(diag(S10),V10));

#X20 reconstruction
U20=U[:,:20];
V20=V[:20,:];
S20=S[:20];
X20=dot(U20,dot(diag(S20),V20));

#X50 reconstruction
U50=U[:,:50];
V50=V[:50:];
S50=S[:50];
X50=dot(U50,dot(diag(S50),V50));

#X100 reconstruction
U100=U[:,:100];
V100=V[:100,:];
S100=S[:100];
X100=dot(U100,dot(diag(S100),V100));

#X200 reconstruction
U200=U[:,:200];
V200=V[:200,:];
S200=S[:200];
X200=dot(U200,dot(diag(S200),V200));


#=========================================================================



#================= ADD YOUR CODE HERE ====================================
# Error of approximation
## TODO: Compute and print the error of each low rank approximation of the matrix
# The Frobenius error can be computed as |X - X_k| / |X|
# Errk where k = [10, 20, 50, 100, 200]


#X10 error
Err10=linalg.norm(X-X10)/linalg.norm(X);
print Err10;

#X20 error
Err20=linalg.norm(X-X20)/linalg.norm(X);
print Err20;

#X50 error
Err50=linalg.norm(X-X20)/linalg.norm(X);
print Err50;

#X100 error
Err100=linalg.norm(X-X100)/linalg.norm(X);
print Err100;

#X200 error
Err200=linalg.norm(X-X200)/linalg.norm(X);
print Err200;



#=========================================================================



# Plot the optimal rank-k approximation for various values of k)
# Create a figure with 6 subfigures
plt.figure(2)

# Rank 10 approximation
plt.subplot(321)
plt.imshow(X10,cmap = cm.Greys_r)
plt.title('Best rank' + str(5) + ' approximation')
plt.axis('off')

# Rank 20 approximation
plt.subplot(322)
plt.imshow(X20,cmap = cm.Greys_r)
plt.title('Best rank' + str(20) + ' approximation')
plt.axis('off')

# Rank 50 approximation
plt.subplot(323)
plt.imshow(X50,cmap = cm.Greys_r)
plt.title('Best rank' + str(50) + ' approximation')
plt.axis('off')

# Rank 100 approximation
plt.subplot(324)
plt.imshow(X100,cmap = cm.Greys_r)
plt.title('Best rank' + str(100) + ' approximation')
plt.axis('off')

# Rank 200 approximation
plt.subplot(325)
plt.imshow(X200,cmap = cm.Greys_r)
plt.title('Best rank' + str(200) + ' approximation')
plt.axis('off')

# Original
plt.subplot(326)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (Rank 480)')
plt.axis('off')

plt.draw()
plt.savefig('best_per_rank.pdf')

#================= ADD YOUR CODE HERE ====================================
# Plot the singular values of the original matrix
## TODO: Plot the singular values of X versus their rank k

plt.figure(3)
plt.plot(linspace(1,S.size,S.size),S)
plt.draw();
plt.ylabel('Singular value')
plt.xlabel("rank")
plt.savefig('svalues.pdf')


plt.figure(4)
SC=cumsum(S)/sum(S)
plt.plot(linspace(1,SC.size,SC.size),SC)
plt.draw();
plt.ylabel('Accumulated energy %')
plt.xlabel("rank")
plt.ylim(0,1)
plt.axhline(y=0.9,color='k',linestyle='--')
plt.savefig('energy.pdf')


#=========================================================================

plt.show() 

