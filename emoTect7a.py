# threholding, Time Sampled, graph, face detection improved
# accuracy improved 87% modified fer only (not self generated database)
import os
import glob
import cv2
import time
import imutils
import numpy as np

from tkinter import *
import tkinter as tk
from tkinter import filedialog
from pandas import DataFrame
from keras.models import load_model
from keras.preprocessing.image import img_to_array

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# loading models for face detection and emotion classification
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
#emotion_classifier = load_model('_mini_XCEPTION.80-0.88.hdf5', compile=False)
emotion_classifier = load_model('_mini_XCEPTION.99-0.66.hdf5', compile=False)

EMOTIONS = ["Angry" , "Disgusted", "Scared", "Happy", "Sad", "Surprised", "Neutral"]
t_pass=[]
frm=0
width = 48
height = 48
dim = (width, height)

e_pass=[[],[],[],[],[],[],[]]
e_count = [0,0,0,0,0,0,0]
e_pcnt = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
undfnd=0 # undefined frames with max prob < 0.20------------------------------------------

e4000_freq = [0,0,0,0,0,0,0] 
e4000_duration = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]

e500_freq = [0,0,0,0,0,0,0] 
e500_duration = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]

e150_freq = [0,0,0,0,0,0,0] 
e150_duration = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
e150_cnt=0 #count of micro expressions.....
micro_folder = r'micro_gen'
micro_csv = r'micro_gen\temp.csv'
files = glob.glob(r'micro_gen\*.jpg')
for f in files:
    os.remove(f)
    
micro_data=pd.read_csv(micro_csv)
for i in range(0,len(micro_data)):
    micro_data.drop(i, inplace=True)

e00_freq = [0,0,0,0,0,0,0] 
e00_duration = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]

prev_label = ""
k=0
nr_of_frames=0
main_option=1
fps = 0.0
camera = cv2.VideoCapture(0)

#Beginning GUI
gui = Tk(className=' EmoTect')
# set window size and position it at center of screen
windowWidth=800
windowHeight=400
positionRight = int(gui.winfo_screenwidth()/2 - windowWidth/2)
positionDown = int(gui.winfo_screenheight()/2 - windowHeight/2)
gui.geometry("{}x{}+{}+{}".format(windowWidth,windowHeight,positionRight,positionDown))
xx=gui.winfo_screenwidth()/2

#gui.geometry("800x400+200+150")#width x heigth
w = Label(gui, text="\nWelcome! \n\nThis tool helps to analyze facial emotions from video clips\n",font=("Helvetica", 15))
w.pack()

v = IntVar()# identifies which one is selected

Label(gui, text="Select one of the following ways of capturing a video:",justify = LEFT,padx = 20).pack()
Radiobutton(gui, text="Real-time Analysis via webcam",padx = 20, variable=v, value=1).pack(anchor=W)
Radiobutton(gui, text="Analysis of a pre-recorded video",padx = 20, variable=v, value=2).pack(anchor=W)

def helloCallBack():
    global fps
    global camera
    global nr_of_frames
    global main_option
    if v.get()==1:
        gui.destroy()
        tempp=Tk(className=' Note')
        # set window size and position it at center of screen
        winWidth=400
        winHeight=200
        posRight = int(tempp.winfo_screenwidth()/2 - winWidth/2)
        posDown = int(tempp.winfo_screenheight()/2 - winHeight/2)
        tempp.geometry("{}x{}+{}+{}".format(winWidth,winHeight,posRight,posDown))
        #tempp.geometry("400x200+400+300")
        Label(tempp,text="\nCallibrating webCam...Please Wait\nThis will take about 15 seconds\n",font=("Helvetica", 10)).pack()
        Label(tempp,text="Press the button below to begin!\n",font=("Helvetica", 10)).pack()
        B1 = Button(tempp, text="Callibrate", command = tempp.destroy)
        B1.pack()
        tempp.mainloop()
        
        # Start default camera
        video = cv2.VideoCapture(0)
        # Number of frames to capture
        num_frames = 350
        
        # Start time
        start = time.time()
        # Grab a few frames
        for i in range(num_frames):
            ret, frame = video.read()
        # End time
        end = time.time()
 
        seconds = end - start # time elapsed
        fps  = num_frames / seconds

        video.release()
        tempp=Tk(className=' Note')
        # set window size and position it at center of screen
        winWidth=400
        winHeight=200
        posRight = int(tempp.winfo_screenwidth()/2 - winWidth/2)
        posDown = int(tempp.winfo_screenheight()/2 - winHeight/2)
        tempp.geometry("{}x{}+{}+{}".format(winWidth,winHeight,posRight,posDown))
        #tempp.geometry("400x200+400+300")
        Label(tempp,text="\nWebCam Callibration Complete\n",font=("Helvetica", 10)).pack()
        Label(tempp,text="Press the button below to begin Real-time streaming!",font=("Helvetica", 10)).pack()
        Label(tempp,text="(Press q to stop recording anytime you wish)\n",font=("Helvetica", 10)).pack()
        B1 = Button(tempp, text="START", command = tempp.destroy)
        B1.pack()
        tempp.mainloop()
        
    if v.get()==2:
        root = Tk(className=' Choose Video...')
        root.geometry("500x100+10+10")#width x heigth
        w1 = Label(root, text="\nBrowse your system for the Test Video...",font=("Helvetica", 15))
        w1.pack()
        root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("All files","*.*"),("jpeg files","*.jpg")))
        test_video_path = root.filename
        root.destroy()
        camera = cv2.VideoCapture(test_video_path)# from the pre recorded video in path
        nr_of_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))# calculate total no of frames
        fps = camera.get(cv2.CAP_PROP_FPS)# calculate fps
        main_option=2
        gui.destroy()
        
        tempp=Tk(className=' Note')
        # set window size and position it at center of screen
        winWidth=400
        winHeight=200
        posRight = int(tempp.winfo_screenwidth()/2 - winWidth/2)
        posDown = int(tempp.winfo_screenheight()/2 - winHeight/2)
        tempp.geometry("{}x{}+{}+{}".format(winWidth,winHeight,posRight,posDown))
        #tempp.geometry("400x200+400+300")
        Label(tempp,text="\nPreliminary Callibration Complete\n",font=("Helvetica", 10)).pack()
        Label(tempp,text="Press the button below to begin video analysis!",font=("Helvetica", 10)).pack()
        Label(tempp,text="(Press q to stop anytime you wish)\n",font=("Helvetica", 10)).pack()
        B1 = Button(tempp, text="START", command = tempp.destroy)
        B1.pack()
        tempp.mainloop()
        
button = Button(gui, text='Confirm', width=25, command=helloCallBack)
button.pack()

gui.mainloop()

# starting video streaming
cv2.namedWindow('TestVideo')
cv2.moveWindow('TestVideo', int(xx-400),75)# width wise centerscreen

# function called by trackbar, sets the next frame to be read
def getFrame(frame_nr):
    global camera
    camera.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)

if main_option==2:
    cv2.createTrackbar("Frames", "TestVideo", 0,nr_of_frames,getFrame)

# font 
font = cv2.FONT_HERSHEY_SIMPLEX  
# fontScale 
fontScale = 0.7
# Blue color in BGR 
color = (255, 0, 0) 
# Line thickness of 1 px 
thickness = 2
#------------------------------------------------------------------------------------------------------------------
while camera.isOpened():
    ret, frame = camera.read()# by default the webcam reads at around 30fps, can be changed by other codes
    if ret==False:
        break
    #reading the frame
    frame = imutils.resize(frame,width=800)
    if main_option==2:
        t_stamp = int(camera.get(cv2.CAP_PROP_POS_FRAMES)/fps)
        if t_stamp%10==0 and ((t_stamp+11)*fps<nr_of_frames) and t_stamp!=0: # 10s time sampling MARCH 15------------------------
            t_stamp=t_stamp+11
            camera.set(cv2.CAP_PROP_POS_FRAMES, int(camera.get(cv2.CAP_PROP_POS_FRAMES))+int(fps*11))
            
        frame = cv2.putText(frame, 'T={0}secs'.format(t_stamp), (10, 30), font, fontScale, color, thickness, cv2.LINE_AA)
        #cv2.imshow("Video", frame)
        # update slider position on trackbar
        # NOTE: this is an expensive operation, remove to greatly increase max playback speed
        cv2.setTrackbarPos("Frames","TestVideo", int(camera.get(cv2.CAP_PROP_POS_FRAMES)))
    
    frame = cv2.putText(frame, 'Press Q to stop',(500, 30), font, 0.7, color, 2, cv2.LINE_AA)
    
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
 
    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    canvas = np.zeros((252, 314, 3), dtype="uint8")
    frameClone = frame.copy()
    
    if detections[0, 0, 0, 2] > 0.75: # 75% confidence of a face existing in the frame
        # compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (fX, fY, fW, fH) = (startX, startY, endX-startX, endY-startY)
        
        crop_face = frame[startY:endY, startX:endX]
        try:
            resized_face = cv2.resize(crop_face, dim, interpolation = cv2.INTER_AREA)
        except Exception as e:
            continue
        gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
        
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare the ROI for classification via the CNN
        roi = gray_face
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        
        if preds[preds.argmax()]<=0.20 and main_option==2: # 0.2----------------------
            undfnd=undfnd+1
            continue
               
        label = EMOTIONS[preds.argmax()]
        
        #print("A D Sc H Sa Su N: ",end="")
        #for i in range(7):
            #print(round(preds[i]*100,2),end ="  ")
        
        frm=frm+1
        t_pass.append(frm) # Plotting graph-----------------------
        for i in range(7):
            if label==EMOTIONS[i]:
                e_count[i]=e_count[i]+1
                e_pass[i].append(e_count[i])
            else:
                e_pass[i].append(e_count[i])
                
                
        if prev_label=="":
            prev_label=label
        else:
            if label==prev_label:
                k=k+1
                prev_label=label
            else:
                if k>=4*fps:#variation after 4s
                    for i in range(7):
                        if prev_label==EMOTIONS[i]:
                            e4000_freq[i]=e4000_freq[i]+1
                            e4000_duration[i]=e4000_duration[i]+k/fps
                elif k>=0.5*fps:#variation after 500ms
                    for i in range(7):
                        if prev_label==EMOTIONS[i]:
                            e500_freq[i]=e500_freq[i]+1
                            e500_duration[i]=e500_duration[i]+k/fps
                elif k>=0.15*fps:#variation after 150ms
                    for i in range(7):
                        if prev_label==EMOTIONS[i]:
                            e150_freq[i]=e150_freq[i]+1
                            e150_duration[i]=e150_duration[i]+k/fps
                            e150_cnt=e150_cnt+1
                            #save prev frame for validation with time stamp
                            micro_file =  os.path.join(micro_folder , str(e150_cnt)+'_'+prev_label+'.jpg')
                            cv2.imwrite(micro_file, prevFrame)
                            end = int(camera.get(cv2.CAP_PROP_POS_FRAMES))-1
                            start = end - k
                            df = pd.DataFrame([[start,end,EMOTIONS[i]]], columns=micro_data.columns)
                            micro_data=pd.concat([micro_data,df])
                else:
                    for i in range(7):
                        if prev_label==EMOTIONS[i]:
                            e00_freq[i]=e00_freq[i]+1
                            e00_duration[i]=e00_duration[i]+k/fps
                k=0
                prev_label=label
        
    else:
        continue
 
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            text = "{}: {:.2f} %".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 7),(7+w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 24),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255, 255, 255), 1)
            cv2.putText(frameClone, label, (fX, fY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)

    prevFrame = frame.copy()#---------------------------------------12 april
    cv2.imshow("Probabilities", canvas)
    cv2.moveWindow("Probabilities", 0,0)
    cv2.imshow('TestVideo', frameClone)
    if cv2.waitKey(1) & 0xFF == ord('q'):# press q to stop
        break

sum = e_count[0]+e_count[1]+e_count[2]+e_count[3]+e_count[4]+e_count[5]+e_count[6]
for i in range(7):
    e_pcnt[i]=round(e_count[i]*100.0/sum,2)

camera.release()
cv2.destroyAllWindows()

def exportData():
    root = Tk(className=' Choose File...')
    root.geometry("500x100+10+10")#width x heigth
    w1 = Label(root, text="\nBrowse your system for the file to export to...",font=("Helvetica", 15))
    w1.pack()
    root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("xlsx files","*.xlsx"),("All files","*.*")))
    test_file_path = root.filename
    root.destroy()
    emoDetect = {'Emotions' : ['Angry' , 'Disgusted', 'Scared', 'Happy', 'Sad', 'Surprised', 'Neutral'],
                 '%':[e_pcnt[0],e_pcnt[1],e_pcnt[2],e_pcnt[3],e_pcnt[4],e_pcnt[5],e_pcnt[6]],
            '0-150ms frequency':[e00_freq[0],e00_freq[1],e00_freq[2],e00_freq[3],e00_freq[4],e00_freq[5],e00_freq[6]],
            '0-150ms duration':[e00_duration[0],e00_duration[1],e00_duration[2],e00_duration[3],e00_duration[4],e00_duration[5],e00_duration[6]],
            '150-500ms frequency':[e150_freq[0],e150_freq[1],e150_freq[2],e150_freq[3],e150_freq[4],e150_freq[5],e150_freq[6]],
            '150-500ms duration':[e150_duration[0],e150_duration[1],e150_duration[2],e150_duration[3],e150_duration[4],e150_duration[5],e150_duration[6]],
            '0.5-4s frequency':[e500_freq[0],e500_freq[1],e500_freq[2],e500_freq[3],e500_freq[4],e500_freq[5],e500_freq[6]],
            '0.5-4s duration':[e500_duration[0],e500_duration[1],e500_duration[2],e500_duration[3],e500_duration[4],e500_duration[5],e500_duration[6]],
            '4s+ frequency':[e4000_freq[0],e4000_freq[1],e4000_freq[2],e4000_freq[3],e4000_freq[4],e4000_freq[5],e4000_freq[6]],
            '4s+ duration':[e4000_duration[0],e4000_duration[1],e4000_duration[2],e4000_duration[3],e4000_duration[4],e4000_duration[5],e4000_duration[6]]
        }
    df = DataFrame(emoDetect, columns= ['Emotions', '%','0-150ms frequency','0-150ms duration','150-500ms frequency','150-500ms duration',
                                       '0.5-4s frequency','0.5-4s duration','4s+ frequency','4s+ duration'])
    df.to_excel (test_file_path, index = None, header=True)
    
    tempp=Tk(className=' Note')
    # set window size and position it at center of screen
    winWidth=400
    winHeight=200
    posRight = int(tempp.winfo_screenwidth()/2 - winWidth/2)
    posDown = int(tempp.winfo_screenheight()/2 - winHeight/2)
    tempp.geometry("{}x{}+{}+{}".format(winWidth,winHeight,posRight,posDown))
    #tempp.geometry("400x200+400+300")
    Label(tempp,text="\n\nData Exported Successfully\n",font=("Helvetica", 10)).pack()
    B1 = Button(tempp, text="Back", command = tempp.destroy)
    B1.pack()
    tempp.mainloop()

# Results -----------------------------------------------
t_passn=np.array(t_pass)
t_passn=100*t_passn/t_pass[-1]
e_passn=np.array(e_pass)
e_passn=100*e_passn/t_pass[-1]

data2 = {'Time': t_passn,
         'Angry': e_passn[0],
         'Disgusted': e_passn[1],
         'Scared': e_passn[2],
         'Happy': e_passn[3],
         'Sad': e_passn[4],
         'Surprised': e_passn[5],
         'Neutral': e_passn[6]
         
        }
df2 = DataFrame(data2,columns=['Time','Angry','Disgusted','Scared','Happy','Sad','Surprised','Neutral'])
    
res = Tk(className=' Final Results')
#res.iconbitmap('colm.ico')
# set window size and position it at center of screen
winWidth=900
winHeight=550
posRight = int(res.winfo_screenwidth()/2 - winWidth/2)
posDown = int(res.winfo_screenheight()/2 - winHeight/2)
res.geometry("{}x{}+{}+{}".format(winWidth,winHeight,posRight,posDown))
#res.geometry("900x550+200+100")
Label(res, text="           The final results are as follows:",font=("Helvetica", 20)).grid(row=0)
Label(res, text="Total Frame Iterations = %d\n" % sum).grid(row =1)
for i in range(4):
    Label(res, text="{0}: {1}%".format(EMOTIONS[i],e_pcnt[i])).grid(row=5*i+2, column=0)
    a1=Label(res, text="0-150ms: {0} times for a total of {1} seconds      ".format(e00_freq[i],round(e00_duration[i],2)))
    a1.grid(row=5*i+3, column=0)
    a2=Label(res, text="150ms-500ms: {0} times for a total of {1} seconds      ".format(e150_freq[i],round(e150_duration[i],2)))
    a2.grid(row=5*i+4, column=0)
    a3=Label(res, text="500ms-4s: {0} times for a total of {1} seconds      ".format(e500_freq[i],round(e500_duration[i],2)))
    a3.grid(row=5*i+5, column=0)
    a4=Label(res, text="4s and above: {0} times for a total of {1} seconds      ".format(e4000_freq[i],round(e4000_duration[i],2)))
    a4.grid(row=5*i+6, column=0)
    Label(res, text="      ").grid(row=5*i+7, column=0)
for i in range(4,7):
    Label(res, text="{0}: {1}%".format(EMOTIONS[i],round(e_count[i]*100.0/sum,3))).grid(row=5*(i-4)+2, column=1)
    a1=Label(res, text="0-150ms: {0} times for a total of {1} seconds      ".format(e00_freq[i],round(e00_duration[i],2)))
    a1.grid(row=5*(i-4)+3, column=1)
    a2=Label(res, text="150ms-500ms: {0} times for a total of {1} seconds      ".format(e150_freq[i],round(e150_duration[i],2)))
    a2.grid(row=5*(i-4)+4, column=1)
    a3=Label(res, text="500ms-4s: {0} times for a total of {1} seconds      ".format(e500_freq[i],round(e500_duration[i],2)))
    a3.grid(row=5*(i-4)+5, column=1)
    a4=Label(res, text="4s and above: {0} times for a total of {1} seconds      ".format(e4000_freq[i],round(e4000_duration[i],2)))
    a4.grid(row=5*(i-4)+6, column=1)
    Label(res, text="      ").grid(row=5*(i-4)+7, column=1)
    
if main_option==2:
    Label(res, text="Undefined : {0}%".format(round(undfnd/nr_of_frames,3))).grid(row=18, column=1)#-------------------------------    
button = Button(res, text='Export Results', width=25, command=exportData)
button.grid(row=0,column=1)

toplevel = tk.Toplevel(res)
toplevel.title("Graphs"),
        
figure2 = plt.Figure(figsize=(10,7), dpi=100)
ax2 = figure2.add_subplot(111)
line2 = FigureCanvasTkAgg(figure2, toplevel)# using toplevel for graph
line2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
df2 = df2[['Time','Angry','Disgusted','Scared','Happy','Sad','Surprised','Neutral']].groupby('Time').sum()
df2.plot(kind='line', legend=True, ax=ax2,fontsize=10)
ax2.set_title('Variation of Emotions over captured frames')

res.mainloop()
#micro_data.drop(0, inplace=True)
micro_data.to_csv(micro_csv,index=False)