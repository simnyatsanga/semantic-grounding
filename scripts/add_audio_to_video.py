import os
from moviepy.editor import VideoFileClip, AudioFileClip

'''
    Add audio segments to their corresponding video segments
    Output:
        - Video segments containing the audio track, saved in MP4 format.
'''
src_video_f = 'data/SpMo_1-1_Day1_Take1_mcp_3DOF_60FPS_handorient.mp4'
src_audio_f = f'data/1-1.wav'
dest_video_f = f'data/SpMo_1-1_Day1_Take1_mcp_3DOF_60FPS_handorient_w_audio.mp4'

# Load the MP4 video file
video_clip = VideoFileClip(src_video_f)

if os.path.isfile(src_audio_f): 
    # Load the audio file
    audio_clip = AudioFileClip(src_audio_f)

    # Set audio to start at the beginning of the video
    # Cut the audio to match the video's length if the audio is longer:
    if audio_clip.duration > video_clip.duration:
        new_audio_clip = audio_clip.subclip(0, video_clip.duration)
        video_clip_with_audio = video_clip.set_audio(new_audio_clip)
    else:
        video_clip_with_audio = video_clip.set_audio(audio_clip)
    
    # Write the result to a new MP4 file
    video_clip_with_audio.write_videofile(dest_video_f, codec="libx264", audio_codec="aac")
else:
    print(f"WARNING: Video clip {src_video_f} does not have an audio file. Saving video clip without.")
    video_clip.write_videofile(dest_video_f, codec="libx264", audio_codec="aac")