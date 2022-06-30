import cv2
import numpy as np
import os
from time import sleep

#--------------------------------the complete code------------------------------
# Get the training classes names and store them in a list
#Here we use folder names for class names

#train_path = 'dataset/train'  # Names are Aeroplane, Bicycle, Car
train_path = '/data-for-knn/data/train'  # Folder Names are Parasitized and Uninfected
training_names = os.listdir(train_path)

# Get path to all images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0

#To make it easy to list all file names in a directory let us define a function
#
def imglist(path):    
    return [os.path.join(path, f) for f in os.listdir(path)]

#Fill the placeholder empty lists with image path, classes, and add class ID number
#
    
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imglist(dir)
    image_paths+=class_path 
    image_classes+=[class_id]*len(class_path)
    class_id+=1

# Create feature extraction and keypoint detector objects
# Create List where all the descriptors will be stored
des_list = list()

sift = cv2.SIFT_create()

for image_path in image_paths:
    im = cv2.imread(image_path)
    im=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    kpts, des = sift.detectAndCompute(im, None)
    des_list.append((image_path,des))


# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for descriptor in des_list:
    descriptors.append(np.vstack(descriptor))
#print(descriptors)
#print(np.vstack(des_list))


#kmeans works only on float, so convert integers to float
#descriptors_float = descriptors.astype(float)  

# Perform k-means clustering and vector quantization
from scipy.cluster.vq import kmeans, vq

k = 10  #k means with 100 clusters gives lower accuracy for the aeroplane example
voc, variance = kmeans(des_list, k, 1) 

# Calculate the histogram of features and represent them as vector
#vq Assigns codes from a code book to observations.
test_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

test_features = stdSlr.transform(test_features)

true_class = [classes_names[i] for i in images_classes]
predictions = [classes_names[i] for i in clf.predict(test_features)]

print("true class="+str(true_class))
print("predict="+str(predictions))

def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('confusion matrix')
    pl.colorbar()
    pl.show()
    
accuracy = accuracy_score(true_class, predictions)
print("accuracy=",accuracy)
cm = confusion_matrix(true_class,predictions)
print(cm)

showconfusionmatrix(cm)
