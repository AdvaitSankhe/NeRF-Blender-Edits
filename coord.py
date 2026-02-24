# import required library
import cv2
import os
# function to display the coordinates of the points clicked on the image
def click_event(event, x, y, flags, params,img_name):
   
   # checking for left mouse clicks
   if event == cv2.EVENT_LBUTTONDOWN:
      print('Left Click')
      print(f'({x},{y})')
 
   # put coordinates as text on the image
   cv2.putText(img, f'({x},{y})', (x, y),   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
   cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
   l1 = []
   l2 = []
   if event == cv2.EVENT_RBUTTONDOWN:
      print('Right Click')
      print(f'({x},{y})')
 
      # put coordinates as text on the image
      cv2.putText(img, f'({x},{y})', (x, y),
      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
      cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
      return [x,y]
# read the input image
for img_name in os.listdir('train/'):
   img = cv2.imread('train/' +img_name)

   # create a window
   cv2.namedWindow('Point Coordinates')

   # bind the callback function to window
   l1 = cv2.setMouseCallback('Point Coordinates', click_event)
   print(l1)
   # display the image
   while True:
      cv2.imshow('Point Coordinates', img)
      k = cv2.waitKey(1) & 0xFF
      if k == 27:
         break
   cv2.destroyAllWindows()
