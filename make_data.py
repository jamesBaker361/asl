#wlasl data from directory -> hf dataser

import PIL
from datasets import load_dataset, Dataset
import json
import os
import cv2

video_dir=os.path.join(os.getcwd(),"videos")
json_path=os.path.join(os.getcwd(),"wlasl.json")

output_dict={
    "label":[],
    "video_cv2":[]
}



with open(json_path) as file:
    j=json.load(file)
    for n,element in enumerate(j):
        label=element["gloss"]
        count=0
        for instance in element["instances"]:
            video_id=instance["video_id"]
            video_path=os.path.join(video_dir,f"{video_id}.mp4")
            bbox=instance["bbox"]
            frame_start=instance["frame_start"]
            frame_end=instance["frame_end"]
            
            if os.path.exists(video_path):
            
                vid = cv2.VideoCapture(video_path)
                
                frame_list=[]

                success =True
                while success:
                    success, image = vid.read() # Read frame
                    if success: 
                        #cv2.imwrite(f"frame{count}.jpg", image) # Save frame
                        #print(image.size)
                        cropped_image=image[bbox[1]:bbox[2],bbox[0]:bbox[2]]
                        #print(cropped_image.size)
                        #cv2.imwrite(f"frame{count}crop.jpg", cropped_image) # Save frame
                        frame_list.append(cropped_image)
                        count += 1
                frame_list=frame_list[frame_start:frame_end]
                output_dict["label"].append(label)
                output_dict["video_cv2"].append(frame_list)
                vid.release()
                print(f"finished video {video_id}")
        Dataset.from_dict(output_dict).push_to_hub("jlbaker361/wlasl")        
Dataset.from_dict(output_dict).push_to_hub("jlbaker361/wlasl")