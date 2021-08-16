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
from tkinter import font as tkFont
#import cv2
#import numpy as np
class App:
     def __init__(self, window, window_title):
         self.window = window
         self.window.title(window_title)

#         self.window.wm_attributes('-alpha', 1.0)
         image=PIL.ImageTk.PhotoImage(PIL.Image.open("image.jpg").resize((1250,600),PIL.Image.LANCZOS))
         
         
         self.video_source=''


         self.images=[]
         
         self.width = 1250
         self.height = 600


                 
         self.canvas = tkinter.Canvas(window, width = self.width, height = self.height)

         self.canvas.create_image(0,0,anchor=NW,image=image)
         self.canvas.pack()

         image1 = PIL.Image.open("replay.png").resize((64,64),PIL.Image.LANCZOS)
         photo1 = PIL.ImageTk.PhotoImage(image1)

         b = tkinter.Button(self.window, image=photo1,command=self.on_click)
         b.place(x=170,y=410)


         
         
         
#         self.canvas.create_rectangle(15, 70, 400, 585, fill="#E6F5FF")
#         self.canvas.create_rectangle(425, 70, 810, 585, fill="#ECECEC")
#         self.canvas.create_rectangle(835, 70, 1220, 585, fill="#DEFFD5")
#         self.canvas.create_rectangle(15, 5, 1220, 55, fill="#024B76")

         self.create_rectangle(15,70, 400,585, outline = 'white',fill='white',width = 3, alpha = 0.3)
         self.create_rectangle(467,70, 771,585, outline = 'white',fill='white',width = 3, alpha = 0.3)
         self.create_rectangle(835,70, 1220,585, outline = 'white',fill='white',width = 3, alpha = 0.3)
         self.create_rectangle(15,5, 1220,55, outline = 'white',fill='black',width = 3, alpha = 0.7)
         
         self.create_rectangle(46, 105, 374, 355, width=2,outline='white',fill='black', alpha = 0.4)

         self.create_rectangle(495, 105, 745, 355, width=2,outline='white',fill='black', alpha = 0.4)

         self.create_rectangle(866, 105, 1194, 200, width=2,outline='white',fill='black', alpha = 0.4)

         fontStyle = tkFont.Font(family="Calibri", size=20,weight='bold')


         ll = tkinter.Label(self.window, text='GAIT SIMULATOR',font=fontStyle,bg='#282828',fg='white')
         ll.place(x=600,y=30, anchor='center')





         self.gei=0
         self.gei_32=0
          
         self.delay = 10
         self.browse()
         
         
         self.window.mainloop()

     def on_click(self,event=None):
         self.vid = MyVideoCapture(self.video_source)
         self.update()

         
     def create_rectangle(self,x1, y1, x2, y2, **kwargs):
       if 'alpha' in kwargs:
          alpha = int(kwargs.pop('alpha') * 255)
          fill = kwargs.pop('fill')
          fill = self.window.winfo_rgb(fill) + (alpha,)
          image = PIL.Image.new('RGBA', (x2-x1, y2-y1), fill)
          self.images.append(PIL.ImageTk.PhotoImage(image))
          self.canvas.create_image(x1, y1, image=self.images[-1], 
                                      anchor='nw')
       self.canvas.create_rectangle(x1, y1, x2, y2, **kwargs)

       
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
        
     def browse(self):
         #self.btn_browse=tkinter.Button(self.window, text="Upload", width=10, command=self.fileDialog)
         #self.btn_browse.pack(anchor=tkinter.W)
         #self.btn_preprocess=tkinter.Button(self.window, text="preprocess", width=25, command=self.preprocessing)
         #self.btn_preprocess.pack(pady=50)

          
         helv36 = tkFont.Font(family='Helvetica', size=10, weight='bold')
         button1 = tkinter.Button(text = "UPLOAD",command=self.fileDialog,fg='white',bg='#0A3E66')
         button1.configure(width = 25, height=2,activebackground = "#D2D2D2", relief = GROOVE)
         button1['font'] = helv36
         button1_window = self.canvas.create_window(100, 525, anchor=NW, window=button1)
         button1.update()
         button2 = tkinter.Button(text = "PREPROCESS",command=self.preprocessing,fg='white',bg='#953B0D')
         button2.configure(width = 25,height=2, activebackground = "#D2D2D2", relief = GROOVE)
         button2['font'] = helv36
         button2_window = self.canvas.create_window(515, 525, anchor=NW, window=button2)
         button2.update()
         button3 = tkinter.Button(text = "PREDICT",command=self.prediction,fg='white',bg='#1D6E09')
         button3.configure(width = 25,height=2, activebackground = "#D2D2D2", relief = GROOVE)
         button3['font'] = helv36
         button3_window = self.canvas.create_window(925, 525, anchor=NW, window=button3)
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
             self.created=self.canvas.create_image(50,110, image = self.photo, anchor = tkinter.NW)
         else:
#             self.canvas.delete(self.created)
             return
 
         self.window.after(self.delay, self.update)
     def preprocessing(self):
         a=self.video_source[-7:-4]
         b=self.video_source[-13:-8]
         c=self.video_source[-17:-14]
         d=self.video_source[-17:-4]
         img_path = 'F:/minor project/code files/code/GEI_CASIA_BB/'+c+'/'+b+'/'+d+'.png'
         print(len(img_path))
         print(img_path)
         self.gei=cv2.imread(img_path)
         print(np.array(self.gei).shape)

         self.gei = cv2.cvtColor(self.gei, cv2.COLOR_BGR2GRAY)
         self.gei_32=cv2.resize(self.gei, (32, 32))/255
#         cv2.namedWindow('GEI', flags=0)
#         cv2.imshow('GEI', self.gei)
#         cv2.waitKey(0)
#         print("GEI.shape: ",self.gei.shape)
#         cv2.destroyAllWindows()

         
         self.photo_gei = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.gei))
         self.canvas.create_image(740,110, image = self.photo_gei, anchor = tkinter.NE)

       
     def prediction(self):
          gei_resized=self.gei_32.reshape(1,32,32,1)
          my_model = tf.keras.models.load_model('F:/minor project/code files/code/models/mymodel_bg_cl_nm_13456/mymodel_gait_bg_cl_nm_13456.h5')
          classification=my_model.predict(gei_resized)
          a=np.argmax(classification[0])
#          print("the predicted value is {}".format(a))
          b='Subject id is '+str(a)
          fontStyle = tkFont.Font(family="Dark Seed", size=25)
          ll = tkinter.Label(self.window, text=b,font=fontStyle,bg='#282828',fg='white')
          ll.place(x=1029,y=150, anchor='center')
         
          

     
              
      
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
