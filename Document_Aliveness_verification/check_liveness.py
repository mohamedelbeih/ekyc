import cv2
import math , sys
# from inference import *

# check up
def check_tilting(pixels,width, height,h_left,h_right,w_bottom,w_up):
    tilting_type = False
    tilting_angle = 10
    sin_tilting_angle = math.sin(math.radians(tilting_angle))
    # if w_up >= ( width + ( 2 * height * sin_tilting_angle ) ):
    if (w_up > w_bottom + pixels) : #and (h_left == h_right):
        tilting_type = 'Up'
    # elif w_bottom >= ( width + ( 2 * height * sin_tilting_angle ) ):
    elif (w_up +pixels < w_bottom) : #and (h_left == h_right):
        tilting_type = 'Down'
    # elif h_right >= ( height + ( 2 * width * sin_tilting_angle ) ):
    elif  (h_left +pixels < h_right): #(w_up == w_bottom) and
        tilting_type = 'Right'
    # elif h_left >= ( height + ( 2 * width * sin_tilting_angle ) ):
    elif (h_left > h_right+pixels): #(w_up == w_bottom) and 
        tilting_type = 'Left'
    else:
        pass
    return tilting_type


#main
# # model = initialze_scripted_model('v.11/model.ts')
# in_file = sys.argv[1]
# print('>>>>>>>>>>>>>>>>',in_file)

# cap = cv2.VideoCapture(in_file)
# count = 0
# detected_tilting = []
# while cap.isOpened():
#     ret,img = cap.read()
#     if not ret: break # break if cannot receive frame
#     card , corner_pts = detect_document(img, model)
#     if card:# replaceing with if card is detected
#         h_right = corner_pts[3][1] -corner_pts[0][1] 
#         h_left = corner_pts[2][1] -corner_pts[1][1] 
#         w_up = corner_pts[1][0] -corner_pts[0][0] 
#         w_bottom = corner_pts[2][0] -corner_pts[3][0] 
#         print(count)
#         count += 1
#         print(h_left,h_left,w_bottom,w_up)
#         # img = cv2.polylines(img, [corner_pts.astype('int')],True, (0, 0, 255), 2) 
#         if abs(h_left - h_right) == 0 and abs(w_up-w_bottom) ==0 :
#             print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
#             global Centered
#             Centered = True
#             img = cv2.polylines(img, [corner_pts.astype('int')],True, (0, 255, 0), 2)
#             global height
#             height = img.shape[0] #card height
#             global width
#             width = img.shape[1] # card width
#             img = cv2.putText(img, 'Centered', (300,444), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
#         try:
#             if Centered :
#                 print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

#                 tilting_type = check_tilting(width, height,h_left,h_right,w_bottom,w_up)
#                 if tilting_type:
#                     detected_tilting.append(tilting_type)
#         except:
#             pass

#     else:
#         # img = cv2.polylines(img, [corner_pts.astype('int')],True, (0, 0 , 255), 2)
#         pass
#     # cv2.imwrite('Output/'+str(count)+".jpg",img)

#     print(">>>>>>>>>>>>>>>>>>>>>>",detected_tilting)    
        
# cap.release()

# with open(in_file.replace("Input","Output").replace('.mp4','.txt'),"w") as f:
#     f.write(str("_".join(list(set(detected_tilting)))))


    























