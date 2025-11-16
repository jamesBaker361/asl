# chip -> hf

import os
import requests
import re
import glob
import json
import cv2
from datasets import Dataset

API_KEY = "AIzaSyBPmYucwLc1zkMqaqfUV1eqGm21PgINzR4"

base_dir="videos"

def get_video_ids(b_dir:str=base_dir):
    youtube_ids=[s for s in os.listdir(b_dir) if os.path.isdir(os.path.join(b_dir,s))]
    output_dict={
        "video_cv2":[],
        "label":[]
    }
    for y in youtube_ids:
        label_list,video_cv2_list=split(y)
        
        output_dict["video_cv2"]+=video_cv2_list
        output_dict["label"]+=label_list
        
        Dataset.from_dict(output_dict).push_to_hub("jlbaker361/youtube-asl")
        
        
    


def split(youtube_id:str):
    print(f"splittign {youtube_id}")
    label_list=[]
    video_cv2_list=[]
    output_path=os.path.join(base_dir,youtube_id)
    with open(os.path.join(output_path,"info.json")) as file:
        json_object=json.load(file)
    desc=json_object["desc"]
    fps=json_object["fps"]
    '''
    output_path=os.path.join(base_dir,youtube_id)
    os.makedirs(output_path,exist_ok=True)
    url=f"https://www.youtube.com/watch?v={youtube_id}"
    download_single_video(url,output_path)
    ydl_opts = {'quiet': True, 'skip_download': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        fps = info.get("fps")
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={youtube_id}&key={API_KEY}"
    data = requests.get(url).json()

    # Get description
    desc = data["items"][0]["snippet"]["description"]'''

    # Regex pattern to capture timestamp and following text
    pattern = r"(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–:]?\s*(.*)"
    matches = re.findall(pattern, desc)
    frame_start_list=[]
    text_list=[]
    for ts, text in matches:
        tsplit=[int(t) for t in ts.split(":")]
        n_frames=fps*60*tsplit[0]+fps*tsplit[1]
        frame_start_list.append(n_frames)
        text_list.append(text)
        
    frame_end_list=frame_start_list[1:]+[-1]
    
    video_path=glob.glob(f"{output_path}/*.mp4")[0]
    
    print(video_path)
    
    
    vid = cv2.VideoCapture(video_path)
                
    frame_list=[]

    success =True
    count=0
    while success:
        success, image = vid.read() # Read frame
        if success: 
            #cv2.imwrite(f"frame{count}.jpg", image) # Save frame
            #print(image.size)
            #cropped_image=image[bbox[1]:bbox[2],bbox[0]:bbox[2]]
            #print(cropped_image.size)
            #cv2.imwrite(f"frame{count}crop.jpg", cropped_image) # Save frame
            frame_list.append(image)
            count += 1
    print("count",youtube_id,count)
    for frame_start,frame_end,label in zip(frame_start_list,frame_end_list,text_list):
        _frame_list=frame_list[frame_start:frame_end]
        label_list.append(label)
        video_cv2_list.append(_frame_list)
    vid.release()
    print(f"finished video {youtube_id}")
    return label_list,video_cv2_list
    
if __name__=="__main__":
    get_video_ids()