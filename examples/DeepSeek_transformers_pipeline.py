from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import movie
from moviepy.editor import TextClip, CompositeVideoClip

# Load the models
retrieval_model_name = "facebook/rag-token-nq"
generation_model_name = "gpt-4o"

# Initialize the tokenizer, retriever, and model
tokenizer = RagTokenizer.from_pretrained(retrieval_model_name)
retriever = RagRetriever.from_pretrained(retrieval_model_name, index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained(retrieval_model_name)

# Example input
input_text = "What is the impact of climate change on polar bears?"

# Tokenize the input
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Retrieve documents and generate output
generated_ids = model.generate(input_ids, num_beams=2, num_return_sequences=1)
output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

# Print the output filename
print(output)

# Read output video file
video.read_videofile("output_video.mp4", fps=24) 
