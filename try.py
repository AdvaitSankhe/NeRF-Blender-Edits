from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tiffile as tifi
import torch
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
#image = cv2.imread('r_22.png')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(model = sam,points_per_side=32,pred_iou_thresh=0.86,stability_score_thresh=0.92,min_mask_region_area=0)
sam.to(device=device)
#my_image = open('r_22.png'.rb)
#predictor = SamPredictor(sam)

#predictor.set_image(image)

for img_name in os.listdir('train2/'):
    #image = cv2.imread('train2/' + img_name)
    image = tifi.imread('train2/' + img_name)
    rows,cols = image.shape[0],image.shape[1]
    print(image.shape)
    #image = image[1500:2500,2000:3000]
    #image = image[:rows//4][:cols//4]
    #plt.imshow(image)
    #plt.show()
    print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('uint8')
    #print(image)
    masks = mask_generator.generate(image)
    with open('masks/dicts2/mask2 ' + img_name + '.txt','w') as f:
        f.write(str(masks))
    plt.figure(figsize=(100,100))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig('masks/imgs2/' + img_name + '.png')
    plt.show()

'''
''
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  
#with open('masks.txt','w') as f:
	#f.write(str(masks))

'''
'''
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks, plt.gca())
#show_box(input_box, plt.gca())
plt.axis('off')
plt.show()
print(masks)
'''
'''
predictor = SamPredictor(sam)

input_point = np.array([[550,420 ]])
input_label = np.array([1])

for img_name in os.listdir('train2/'):
	print(img_name)
	image = cv2.imread('train2/' + img_name)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	predictor.set_image(image)
	masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
	)
	with open('masks/dicts2/drummask2 ' + img_name + '.txt','w') as f:
		f.write(str(masks))
	print(masks)
	for i, (mask, score) in enumerate(zip(masks, scores)):
	    plt.figure(figsize=(10,10))
	    plt.imshow(image)
	    show_mask(mask, plt.gca())
	    show_points(input_point, input_label, plt.gca())
	    mask = mask*255
	    mask = np.uint8(mask)
	    print(image.dtype)
	    mask=cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)#change mask to a 3 channel image 
	    mask_out=cv2.subtract(mask,image)
	    mask_out=cv2.subtract(mask,mask_out)
	    cv2.imwrite('masks/imgs2/drumcropped_' +img_name + str(i)+'.png',mask_out)
	    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
	    plt.axis('off')
	    plt.savefig('masks/imgs2/' + img_name + '_idx_' + str(i) + '.png')
	    plt.show()   
'''