from moviepy.editor import TextClip, CompositeVideoClip

# Generate text using the RAG pipeline (as shown above)
generated_text = "Your generated text here."

# Create a text clip
text_clip = TextClip(generated_text, fontsize=24, color='white', size=(640, 480))

# Set the duration of the text clip
text_clip = text_clip.set_duration(10)

# Create a video clip
video = CompositeVideoClip([text_clip])

# Write the video to a file
video.write_videofile("output_video.mp4", fps=24)

