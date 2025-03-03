import os
import gymnasium as gym
import time
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import io
import warnings
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# Our Host URL should not be prepended with "https" nor should it have a trailing slash.
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'

# Sign up for an account at the following link to get an API Key.
# https://beta.dreamstudio.ai/membership

# Click on the following link once you have created an account to be taken to your API Key.
# https://beta.dreamstudio.ai/membership?tab=apiKeys

os.environ['STABILITY_KEY'] = 'sk-7181FQwrpey6rLrFOfHtwTQpZgwsmwwNiPykiAzpJTEtKFah-prev'

# Set up our connection to the API.
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'], # API Key reference.
    verbose=True, # Print debug messages.
    engine="stable-diffusion-v1-5", # Set the engine to use for generation. 
    # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0 
    # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-inpainting-v1-0 stable-inpainting-512-v2-0
)


answers = stability_api.generate(
    # prompt="a mountain landscape in the style of thomas kinkade",
    prompt="An evil laugh with resemblance to Night in Death Note, showing only one person and his face with high saturation and hot color temperature.",
    seed=9080980, # If a seed is provided, the resulting generated image will be deterministic.
                    # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                    # Note: This isn't quite the case for Clip Guided generations, which we'll tackle in a future example notebook.
    steps=30, # Amount of inference steps performed on image generation. Defaults to 30. 
    cfg_scale=8.0, # Influences how strongly your generation is guided to match your prompt.
                   # Setting this value higher increases the strength in which it tries to match your prompt.
                   # Defaults to 7.0 if not specified.
    width=512, # Generation width, defaults to 512 if not included.
    height=512, # Generation height, defaults to 512 if not included.
    samples=1, # Number of images to generate, defaults to 1 if not included.
    sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                 # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                 # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
)

# Set up our warning to print to the console if the adult content classifier is tripped.
# If adult content classifier is not tripped, save generated images.
for resp in answers:
    for artifact in resp.artifacts:
        if artifact.finish_reason == generation.FILTER:
            warnings.warn(
                "Your request activated the API's safety filters and could not be processed."
                "Please modify the prompt and try again.")
        if artifact.type == generation.ARTIFACT_IMAGE:
            img = Image.open(io.BytesIO(artifact.binary))
            img.save(str(artifact.seed)+ ".png") # Save our generated images with their seed number as the filename.


# Note: With multi-prompting, we can mix concepts by assigning each prompt a specific weight. Concepts are combined according to their weight. 
# Prompts with token lengths beyond 77 will be truncated. Default prompt weight is 1 if not specified.
# prompt="An evil laugh with resemblance to Night in Death Note, showing only one person and his face with high saturation and hot color temperature."
answers = stability_api.generate(
    # prompt= [generation.Prompt(text="a mountain landscape",parameters=generation.PromptParameters(weight=1)), 
    # generation.Prompt(text="in the style of thomas kinkade",parameters=generation.PromptParameters(weight=1)), 
    # generation.Prompt(text="tree",parameters=generation.PromptParameters(weight=-1.3))],

    prompt= [generation.Prompt(text="An evil kitty",parameters=generation.PromptParameters(weight=1)), 
    generation.Prompt(text="with resemblance to Night in Death Note, showing only one person and his face with high saturation and hot color temperature",parameters=generation.PromptParameters(weight=1.3)), 
    generation.Prompt(text="kindness",parameters=generation.PromptParameters(weight=-0.5))], 
    # Negative prompting is now possible via the API, simply assign a negative weight to a prompt.
    # In the example above we are combining a mountain landscape with the style of thomas kinkade, and we are negative prompting trees out of the resulting concept.
    # When determining prompt weights, the total possible range is [-10, 10] but we recommend staying within the range of [-2, 2].
    seed=1000080980, # If a seed is provided, the resulting generated image will be deterministic.
                    # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                    # Note: This is only true for non-CLIP Guided generations. 
    steps=30, # Amount of inference steps performed on image generation. Defaults to 30. 
    cfg_scale=8.0, # Influences how strongly your generation is guided to match your prompt.
                   # Setting this value higher increases the strength in which it tries to match your prompt.
                   # Defaults to 7.0 if not specified.
    width=512, # Generation width, defaults to 512 if not included.
    height=512, # Generation height, defaults to 512 if not included.
    samples=1, # Number of images to generate, defaults to 1 if not included.
    sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                 # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                 # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
)

# Set up our warning to print to the console if the adult content classifier is tripped.
# If adult content classifier is not tripped, save generated images.
for resp in answers:
    for artifact in resp.artifacts:
        if artifact.finish_reason == generation.FILTER:
            warnings.warn(
                "Your request activated the API's safety filters and could not be processed."
                "Please modify the prompt and try again.")
        if artifact.type == generation.ARTIFACT_IMAGE:
            img = Image.open(io.BytesIO(artifact.binary))
            img.save(str(artifact.seed)+ ".png") # Save our generated images with their seed number as the filename.

answers = stability_api.generate(
    prompt= [generation.Prompt(text="A professional presentation slide deck containing system and data pipeline architecture",parameters=generation.PromptParameters(weight=2)), 
    generation.Prompt(text="with heading LLM = AI21: Jurassic-1 Grande, 17B parameters + stable diffusion animation model",parameters=generation.PromptParameters(weight=1.3)), 
    generation.Prompt(text="in AWS SageMaker with FSX ONTAP",parameters=generation.PromptParameters(weight=1.5)), 
    generation.Prompt(text="casual, funny",parameters=generation.PromptParameters(weight=-0.5))], 
    # Negative prompting is now possible via the API, simply assign a negative weight to a prompt.
    # In the example above we are combining a mountain landscape with the style of thomas kinkade, and we are negative prompting trees out of the resulting concept.
    # When determining prompt weights, the total possible range is [-10, 10] but we recommend staying within the range of [-2, 2].
    seed=9070970, # If a seed is provided, the resulting generated image will be deterministic.
                    # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                    # Note: This is only true for non-CLIP Guided generations. 
    steps=30, # Amount of inference steps performed on image generation. Defaults to 30. 
    cfg_scale=8.0, # Influences how strongly your generation is guided to match your prompt.
                   # Setting this value higher increases the strength in which it tries to match your prompt.
                   # Defaults to 7.0 if not specified.
    width=512, # Generation width, defaults to 512 if not included.
    height=512, # Generation height, defaults to 512 if not included.
    samples=3, # Number of images to generate, defaults to 1 if not included.
    sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                 # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                 # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
)

# Set up our warning to print to the console if the adult content classifier is tripped.
# If adult content classifier is not tripped, save generated images.
for resp in answers:
    for artifact in resp.artifacts:
        if artifact.finish_reason == generation.FILTER:
            warnings.warn(
                "Your request activated the API's safety filters and could not be processed."
                "Please modify the prompt and try again.")
        if artifact.type == generation.ARTIFACT_IMAGE:
            img = Image.open(io.BytesIO(artifact.binary))
            img.save(str(artifact.seed)+ ".png") # Save our generated images with their seed number as the filename.

# TODO: test image-to-image API; using evil_laugh.png as the input image
# next section is for testing image-to-image API using evil_laugh.png as the input image, use image-to-image API in stability_api, or in generation_api, take previous resp in answers as the input image


# next section is for testing image-to-image API using evil_laugh.png as the input image

# global g_image_rgb  # use a global variable to store an image

# Set up the figure and axis for the animation
global fig_ai, ax_ai, im_ai, raster, im_laugh
fig_ai, ax_ai = plt.figure(), plt.axes()
ax_ai.axis('off')


def open_img(filename='media/evil_laugh.png'):
    """
    open the image and convert it to RGB, default filename is 'media/evil_laugh.png'
    """
    # check if filename is a valid image file
    try:
        img = Image.open(filename)
        image_rgb = img.convert('RGB')
        plt.axis('off')
        raster = plt.imshow(image_rgb)  # raster is a <matplotlib.image.AxesImage object at 0x120414E50>
        # plt.show()
        return image_rgb  # <PIL.Image.Image image mode=RGB size=768x576 at 0x120414E50>
        # return raster # <matplotlib.image.AxesImage object at 0x1247796d0>
    except Exception as e:
        print(f"Error opening image: {e}")
        return None


def add_shoulderbox(image, flicker=False):
    # Define a function to add a box to the left shoulder
    # Get the dimensions of the image
    width, height = image.size

    # set coordinates of the left shoulder and mouth to global & initialize
    global x1_left_shoulder, x2_left_shoulder, y1_left_shoulder, y2_left_shoulder
    # Define the coordinates of the left shoulder
    x1_left_shoulder = int(width * 0.55)
    x2_left_shoulder = int(width * 0.65)
    y1_left_shoulder = int(height * 0.45)
    y2_left_shoulder = int(height * 0.55)

    # Add black pixels to the left shoulder region to simulate a box
    for x in range(x1_left_shoulder, x2_left_shoulder):
        for y in range(y1_left_shoulder, y2_left_shoulder):
            # Flicker effect: randomly change the color in the left shoulder region
            if flicker and random.random() < 0.1:
                image.putpixel((x, y), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            else:
                image.putpixel((x, y), (0, 0, 0))
    return image


global im_box_added, im_laugh
im_tmp = open_img()
im_laugh = add_shoulderbox(im_tmp, flicker=True)


def animate_evil_laugh(frame, flicker=True):
    # Define a function to update the plot for each frame of the animation
    # Assumption: the image is already loaded into g_image_rgb
    im_ai = im_laugh
    #im_ai = ax_ai.imshow(im_box_added)

    # open the image and convert it to RGB, filename is 'media/evil_laugh.png'

    # observation_upt, _, _, info_upt, _ = env.step(env.action_space.sample())
    # Load the laughing face image
    # modify im_laugh?
    # image_ary = im_ai.get_array()
    image_ary = np.array(im_ai)

    # im_laugh = image_tmp
    # image_ary = im_laugh.get_array()
    # im_laugh.set_data(image_ary)

    # Add a box to the left shoulder
    img_shrbx = add_shoulderbox(im_ai, flicker=True)

    # Apply flickering to the shoulder box and mouth:
    image_array = np.array(img_shrbx)  # convert image to numpy array

    # Create a boolean mask to apply flickering
    flicker_mask = np.random.random(image_array.shape[:2]) < 0.2
    # np.random.random returns a read-only array in newer versions of NumPy.
    # To fix: convert the read-only array to a writable array using the np.copy function before modifying it.
    # flicker_mask = np.copy(np.random.random(image_array.shape[:2]) < 0.2)

    mask = np.broadcast_to(flicker_mask[..., np.newaxis], image_array.shape)
    mask = np.array(mask)  # <-- create a new array that is not read-only

    mouth_x1 = int(image_array.shape[1] * 0.2)
    mouth_x2 = int(image_array.shape[1] * 0.8)
    mouth_y1 = int(image_array.shape[0] * 0.6)
    mouth_y2 = int(image_array.shape[0] * 0.7)
    mask[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = True
    flicker_mouth = np.random.random(image_array.shape[:2]) < 0.2
    mask = np.logical_and(mask, flicker_mouth)

    # Set the mask region to a random color
    color = np.random.randint(0, 256, 3)
    color = color.reshape(1, 1, 3)
    mask = mask.reshape(mask.shape[0], mask.shape[1], 1)  # Reshape mask to match color
    image_array[mask] = color
    # image_array[mask[:, :, None]] = color

    # Draw the text bubble with the additional message
    draw = ImageDraw.Draw(img_shrbx)
    bubble_x1 = int(image_array.shape[1] * 0.25)
    bubble_x2 = int(image_array.shape[1] * 0.75)
    bubble_y1 = int(image_array.shape[0] * 0.15)
    bubble_y2 = int(image_array.shape[0] * 0.25)
    draw.rectangle((bubble_x1, bubble_y1, bubble_x2, bubble_y2), fill='white')
    font = ImageFont.truetype("arial.ttf", 24)
    # TODO: add time to the text bubble
    draw.text((bubble_x1 + 10, bubble_y1 + 10), "HAHAHA!! I've become SMARTER!", font=font, fill='black')

    # Convert the numpy array back to an image
    new_image = Image.fromarray(np.uint8(image_array))

    # Update the plot with the new image
    im_laugh.set_data(np.array(new_image))

    # Return the updated plot
    return im_laugh,

