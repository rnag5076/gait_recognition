import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import joblib
import numpy as np
import os
import glob as gb
import tensorflow as tf
class App:
     def __init__(self, window, window_title):
         self.window = window
         self.window.title(window_title)
         self.video_source=''
         
         self.width = 1000
         self.height = 600


         self.gei=0
         
         self.canvas = tkinter.Canvas(window, width = self.width, height = self.height)
         self.canvas.pack()        

         
         
         self.delay = 10
         self.browse()
         
         
         self.window.mainloop()


     def visitDir(self,path):
         if not os.path.isdir(path):
             print('Error: "', path, '" is not a directory or does not exist.')
             return
         else:
             try:
                 for lists in os.listdir(path):
                     sub_path = os.path.join(path, lists)
                     self.num += 1
             except:
                 pass

     
     def pixels_sum(self,image, height, width):
         sum = 0
         for heights in range(height):
             for widths in range(width):
                 if image[heights, widths] == 255:
                     sum +=1
         return sum
     def find_dis_edge_to_head(self,image, height, width):
         flag_of_top = 0
         for heights in range(height):
             for widths in range(width):
                 if flag_of_top == 0 and image[heights, widths] == 255:
                     edge_to_head_f = heights
                     flag_of_top = 1
         return edge_to_head_f
     def find_dis_edge_to_feet(self,image, height, width):
         flag_of_feet = 0
         for heights in range(height):
             for widths in range(width):
                 if flag_of_feet == 0 and image[img.shape[0] - heights, img.shape[1] - widths] == 255:
                     edge_to_feet_f = heights
                     flag_of_feet = 1
         return edge_to_feet_f
     def browse(self):
         #self.btn_browse=tkinter.Button(self.window, text="Upload", width=10, command=self.fileDialog)
         #self.btn_browse.pack(anchor=tkinter.W)
         #self.btn_preprocess=tkinter.Button(self.window, text="preprocess", width=25, command=self.preprocessing)
         #self.btn_preprocess.pack(pady=50)

         button1 = tkinter.Button(text = "Upload",command=self.fileDialog)
         button1.configure(width = 25, activebackground = "#D2D2D2", relief = GROOVE)
         button1_window = self.canvas.create_window(10, 550, anchor=NW, window=button1)
         button1.update()
         button2 = tkinter.Button(text = "preprocess",command=self.preprocessing)
         button2.configure(width = 25, activebackground = "#D2D2D2", relief = GROOVE)
         button2_window = self.canvas.create_window(300, 550, anchor=NW, window=button2)
         button2.update()
         button3 = tkinter.Button(text = "predict",command=self.prediction)
         button3.configure(width = 25, activebackground = "#D2D2D2", relief = GROOVE)
         button3_window = self.canvas.create_window(500, 550, anchor=NW, window=button3)
         button3.update()

     def fileDialog(self):
         self.filename=filedialog.askopenfilename(initialdir="/",title="select a file",filetype=(("avi","*.avi"),("All files","*.*")))
         self.video_source=self.filename
         print(self.filename)
         self.vid = MyVideoCapture(self.video_source)
         self.update()
 
     def update(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()
 
         if ret:
             self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
             created=self.canvas.create_image(50,50, image = self.photo, anchor = tkinter.NW)
         else:
             self.canvas.delete(created)
             return
 
         self.window.after(self.delay, self.update)
     def preprocessing(self):
          img_list = []
          vc = cv2.VideoCapture(self.video_source)
          c = 1

          if vc.isOpened():
              rval, frame = vc.read()
              print("Read Successfully")

          else:
              rval = False

          count = 0
          name_no = 0

          if not os.path.isdir('Frames'):
               os.mkdir('Frames')
          if not os.path.isdir('sil_casia_b'):
               os.mkdir('sil_casia_b')
          while rval:  
              rval, frame = vc.read()
              count += 1
              if count >= 0 & count < 500:
                  if (count % 2 == 0):  # store each 3rd frames
                      name_no += 1
                      if(name_no==56):
                          break
                      img_list.append(frame)
                      cv2.imwrite('Frames/' + str(name_no) + '.png', frame)  # store frames
          vc.release()

          degree=self.video_source[-7:-4]
          subject_number=self.video_source[-17:-14]
          source='F:/minor project/DatasetB/videos/{}-bkgrd-{}.avi'.format(subject_number,degree)
          self.vid1 = MyVideoCapture(source)
          ret, background = self.vid1.get_frame()
          background_bgr = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)


          
          self.num = 0
          path_list = []
          self.visitDir("Frames\\")
          for i in range(self.num + 1):
              path_no = "Frames\\" + str(i) + ".png"
              if i > 0:
                  path_list.append(path_no)
          print(path_list)

          for i in range(self.num):
              path = path_list[i]
              frame = cv2.imread(path)
              print(i)
              
              subtraction_back_frame = cv2.subtract(background_bgr, frame)
              subtraction_frame_back = cv2.subtract(frame, background_bgr)
              subtraction_back_frame = cv2.cvtColor(subtraction_back_frame, cv2.COLOR_BGR2GRAY)  
              subtraction_frame_back = cv2.cvtColor(subtraction_frame_back, cv2.COLOR_BGR2GRAY)
              ret1, silhouette1 = cv2.threshold(subtraction_back_frame, 27, 255, cv2.THRESH_BINARY)
              ret2, silhouette2 = cv2.threshold(subtraction_frame_back, 27 , 255, cv2.THRESH_BINARY)
              binary_silhouette = cv2.bitwise_or(silhouette1, silhouette2) 
              cv2.imwrite('sil_casia_b/' + str(i+1) + '.png', binary_silhouette)
          print('done')


          img_path = gb.glob("sil_casia_b\\*.png")

          binary_silhouette_crop_list = []  # For storing all binary_silhouette frames of one gait video
          cx_list = []  # For storing all x coordinate of centroid
          cx_right_list = []  # For storing all (width-cx)
          edge_to_head_list = []
          edge_to_feet_list = []
          img_list = []
          foreground_pixels_list = []
          gait_period_flag_value_list = []
          gait_period_flag_list = []
          gait_period_flag_plot_list = []
          one_gait_period_img_list = []
          gait_period_flag_real_index_list = []
          count = 0

          for path in img_path:
              count += 1
              if count > 4 and count < (self.num - 4):
                  img = cv2.imread(path)
                  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                  img_list.append(img)
                  foreground_pixels = self.pixels_sum(img, img.shape[0], img.shape[1])
                  foreground_pixels_list.append(foreground_pixels)
          for i in range(len(foreground_pixels_list) - 3):
              if foreground_pixels_list[i + 3] < foreground_pixels_list[i + 2] and foreground_pixels_list[i + 3] < \
                      foreground_pixels_list[i + 1] and foreground_pixels_list[i + 3] < foreground_pixels_list[i + 4] and \
                      foreground_pixels_list[i + 3] < foreground_pixels_list[i + 5]:
                  gait_period_flag_value_list.append(foreground_pixels_list[i + 3])
                  gait_period_flag_list.append(i)
          for i in range(len(gait_period_flag_value_list)):
              gait_period_flag_real_index_list.append(gait_period_flag_list[i] + 3)
          print("gait_period_flag_real_index_list", gait_period_flag_real_index_list)
          print("gait_period_flag_list", gait_period_flag_list)
          one_gait_period_img_list = (img_list[gait_period_flag_list[0] + 3:gait_period_flag_list[2] + 3])
          print("Selected frames: ", len(img_list))
          print("One Gait Period Has: ", len(one_gait_period_img_list), "frames")


          for i in range(len(one_gait_period_img_list)):
              gait_period_frame = one_gait_period_img_list[i]
              m = cv2.moments(gait_period_frame)  # moment
              cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']  # Centroid of binary_silhouette
              cx = int(cx)  # For Normalise & Align
              cx_right = gait_period_frame.shape[1] - cx
              cx_right_list.append(cx_right)
              cx_list.append(cx)
              edge_to_head = self.find_dis_edge_to_head(gait_period_frame, gait_period_frame.shape[0], gait_period_frame.shape[1])
              edge_to_feet = self.find_dis_edge_to_head(gait_period_frame, gait_period_frame.shape[0], gait_period_frame.shape[1])
              edge_to_head_list.append(edge_to_head)
              edge_to_feet_list.append(edge_to_feet)
          cx_left_min = cx_list[cx_list.index(min(cx_list))]
          cx_right_min = cx_right_list[cx_right_list.index(min(cx_right_list))]
          edge_to_head_min = edge_to_head_list[edge_to_head_list.index(min(edge_to_head_list))]
          edge_to_feet_min = edge_to_feet_list[edge_to_feet_list.index(min(edge_to_feet_list))]
          min_dis = min(cx_right_min, cx_left_min)
          print("Minimum distance of all frames from centroid to left edge: ", cx_left_min)
          print("Minimum distance of all frames from centroid to right edge: ", cx_right_min)
          print("Minimum distance of all frames from top dege to subjects' heads: ", edge_to_head_min)
          print("Minimum distance of all frames from bottom dege to subjects' feet: ", edge_to_feet_min)

          for i in range(len(one_gait_period_img_list)):
              cx = cx_list[i]
              gait_period_frame = one_gait_period_img_list[i]
              crop = gait_period_frame[edge_to_head_min:img.shape[0]-edge_to_feet_min+5, cx - min_dis:cx + min_dis]
              crop = crop.astype(np.uint64)
              binary_silhouette_crop_list.append(crop)

          for i in range(len(one_gait_period_img_list)):
              if i == 0:
                  gei_sum = binary_silhouette_crop_list[0]
              if i > 0:
                  gei_sum = cv2.add(gei_sum, binary_silhouette_crop_list[i])
          self.gei = gei_sum / len(one_gait_period_img_list)
          self.gei = self.gei.astype(np.uint8)
#          gei=cv2.resize(gei, (32, 32)) / 255
#          cv2.namedWindow('GEI', flags=0)
#          cv2.imshow('GEI', self.gei)
#          cv2.waitKey(0)
          print("GEI.shape: ",self.gei.shape)
#          cv2.destroyAllWindows()

          self.photo_gei = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.gei))
          self.canvas.create_image(350,350, image = self.photo_gei, anchor = tkinter.NE)


     def prediction(self):
          gei_resized=cv2.resize(self.gei, (32, 32)) / 255
          my_model = tf.keras.models.load_model('F:/minor project/code files/code/models/mymodel_bg_cl_nm_13456/mymodel_gait_bg_cl_nm_13456.h5')
          gei_resized=gei_resized.reshape(1,32,32,1)
          classification=my_model.predict(gei_resized)
          print("the predicted value is {}".format(np.argmax(classification[0])))
         
          

     
              
      
class MyVideoCapture:
     def __init__(self, video_source):
         # Open the video source
         self.vid = cv2.VideoCapture(video_source)
         if not self.vid.isOpened():
             raise ValueError("Unable to open video source", video_source)
 
         # Get video source width and height
         
 
     def get_frame(self):
         if self.vid.isOpened():
             ret, frame = self.vid.read()
             if ret:
                 # Return a boolean success flag and the current frame converted to BGR
                 return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
             else:
                 return (ret, None)
         else:
             return (ret, None)
 
     # Release the video source when the object is destroyed
     def __del__(self):
         if self.vid.isOpened():
             self.vid.release()
 
 # Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")
