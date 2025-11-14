#downloads from youtube -> chip -> hf

from yt_download import download_single_video
import os
import yt_dlp
import requests
import re
import glob
import cv2

API_KEY = "AIzaSyBPmYucwLc1zkMqaqfUV1eqGm21PgINzR4"

base_dir="videos"



def split(youtube_id:str):
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
    desc = data["items"][0]["snippet"]["description"]

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
    print("?")
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
    print("count",count)
'''    frame_list=frame_list[frame_start:frame_end]
    output_dict["label"].append(label)
    output_dict["video_cv2"].append(frame_list)
    vid.release()
    print(f"finished video {video_id}")'''
    
if __name__=="__main__":
    split("dUFIu42tca4")