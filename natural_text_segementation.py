import os
import torch
import torchvision
from PIL import Image
import CLIP.clip as clip
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
import time
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
print("CLIP:", clip)
print("CLIP functions:", dir(clip))

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model():
    """
    Loads the CLIP model.

    Returns:
      The CLIP model and the preprocess function.
    """
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    return clip_model, preprocess


def load_sam_model():
    """
    Loads the SAM 2 model.

    Returns:
      The SAM 2 model and the mask generator.
    """
    sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    sam2 = build_sam2(
        model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False
    )
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    return sam2, mask_generator


def load_gemini_model():
    """
    Loads the Gemini model.

    Returns:
      The Gemini model.
    """
    GEMINI_API_KEY = "AIzaSyAVi-LojQT7143OpXmdogGjAe6yBlwWHSI"

    # Configure Gemini
    genai.configure(api_key=GEMINI_API_KEY)

    gemini_model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        # Create the Gemini model
        generation_config={
            "temperature": 0.1,  # Adjust temperature for more focused responses
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        },
    )
    
    [context, example_cases] = load_gemini_context()
    
    return gemini_model, context, example_cases


def load_gemini_context():
    files = [
        upload_to_gemini("./images/cars.jpg"),
        upload_to_gemini("./images/red_car.png"),
        upload_to_gemini("./images/pink_car_0.png"),
        upload_to_gemini("./images/pink_car_1.png"),
        upload_to_gemini("./images/red_car_wheel.png"),
        upload_to_gemini("./images/road.png"),
    ]

    return [
        [
            "Having this image as a reference, I will send you highlited spots in the image an you will answer with only YES or NO ",
            files[0],
            files[1],
            "Is this a part of a red car?",
            "output: YES or NO YES",
            "Having this image as a reference, I will send you highlited spots in the image an you will answer with only YES or NO ",
            files[0],
            files[1],
            "Is this a part of a red car?",
            "output: YES or NO NO",
            "Having this image as a reference, I will send you highlited spots in the image an you will answer with only YES or NO ",
            files[0],
            files[1],
            "Is this a part of a pink car?",
            "output: YES or NO YES",
            # "Having this image as a reference, I will send you highlited spots in the image an you will answer with only YES or NO ",
            # files[0],
            # files[5],
            # "Is this a part of a road?",
            # "output: YES or NO YES",
            # "Having this image as a reference, I will send you highlited spots in the image an you will answer with only YES or NO ",
            # files[0],
            # files[4],
            # "Is this a part of a road?",
            # "output: YES or NO NO",
            # "Having this image as a reference, I will send you highlited spots in the image an you will answer with only YES or NO ",
            # files[0],
            # files[4],
            # "Is this a part of a red car?",
            # "output: YES or NO YES",
        ],
        3,
    ]


def append_to_gemini_context(context, original, masked, text_prompt):
    # Upload the masked image to Gemini
    new_context = [
        "Having this image as a reference, I will send you highlited spots in the image an you will answer with only YES or NO ",
        original,
        masked,
        f"Is this a part of {text_prompt}?",
        "output: YES or NO ",
    ]

    all_context = context + new_context

    return all_context

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


def upload_masked_to_gemini(image_path, mask, mime_type="image/png"):
    """
    Applies the mask to the image and uploads the result to Gemini.
    """
    # Apply the mask to the image
    image = Image.open(image_path)
    masked_image = image * np.array(mask)[..., None]
    masked_image = Image.fromarray(masked_image.astype("uint8"))

    # Save the masked image temporarily
    temp_file_path = "temp_masked_image.png"
    masked_image.save(temp_file_path)

    # plot masked image
    plt.figure(figsize=(10, 10))
    plt.imshow(masked_image)
    plt.axis("off")
    plt.show()

    # Upload the masked image to Gemini
    file = upload_to_gemini(temp_file_path, mime_type=mime_type)

    # Remove the temporary file
    os.remove(temp_file_path)

    return file


def calculate_image_clip_embedding(image_path):
    """
    Calculates the CLIP embedding for the given image.

    Args:
      image_path: Path to the image.

    Returns:
      A tensor of the CLIP embedding for the image.
    """
    image = Image.open(image_path)
    
    image_array = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image_array)
    return image_features, image_array


def calculate_text_clip_embedding(text):
    """
    Calculates the CLIP embedding for the given text.

    Args:
      text: Text to calculate the embedding for.

    Returns:
      A tensor of the CLIP embedding for the text.
    """
    text_token = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_token)
    return text_features


def calculate_clips_similarity(image_path, generated_masks, text_prompt):
    """
    Calculates the similarity between the text prompt and the generated masks.

    Args:
      image_path: Path to the image.
      generated_masks: A list of masks generated by SAM 2.
      text_prompt: Text prompt describing the object to segment.

    Returns:
      A list of similarities between the text and mask features.
    """
    # Load and preprocess the image
    image = Image.open(image_path)

    # Generate CLIP embedding for the text prompt
    text_features = calculate_text_clip_embedding(text_prompt)

    # Calculate CLIP embeddings for the candidate masks
    mask_features = calculate_clip_embeddings_for_masks(
        generated_masks, np.array(image)
    )

    # Normalize the features and calculate the similarity
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    mask_features = mask_features / mask_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * text_features @ mask_features.T).softmax(dim=-1)
    return similarity[0].tolist()


def calculate_sam_masks(image_path):
    """
    Calculates masks using SAM 2 for the given image.

    Args:
      image_path: Path to the image.

    Returns:
      A list of masks generated by SAM 2.
    """
    image = Image.open(image_path)
    image_array = np.array(image)
    masks = mask_generator.generate(image_array)

    # Extract masks and scores (adapt this based on SAM 2 output format)
    generated_masks = [mask["segmentation"] for mask in masks]
    scores = [mask["stability_score"] for mask in masks]

    return [generated_masks, scores]


def calculate_clip_embeddings_for_masks(masks, image):
    """
    Calculates the CLIP embeddings for the given masks.

    Args:
      masks: A list of masks.

    Returns:
      A tensor of the CLIP embeddings for the masks.
    """
    mask_features = []
    for mask in masks:
        masked_image = image * np.array(mask)[..., None]
        masked_image_array = (
            preprocess(Image.fromarray(masked_image.astype("uint8")))
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            mask_feature = clip_model.encode_image(masked_image_array)
        mask_features.append(mask_feature)
    mask_features = torch.cat(mask_features, dim=0)
    return mask_features


def calculate_gemini_response(image_path, generated_masks, text_prompt):
    """
    Calculates the response from Gemini for the given image and masks.

    Args:
      image_path: Path to the image.
      generated_masks: A list of masks generated by SAM 2.
      text_prompt: Text prompt describing the object to segment.

    Returns:
      The response from Gemini.
    """
    original_file = upload_to_gemini(image_path)

    gemini_responses = []
    counter = 0
    for mask in generated_masks:
        # just for testing purposes
        if counter < 15:
            # Upload masked image to Gemini
            masked_file = upload_masked_to_gemini(image_path, mask)

            # Prepare the prompt for Gemini
            new_context = append_to_gemini_context(
                context, original_file, masked_file, text_prompt
            )

            # Get Gemini's response
            try:
                response = gemini_model.generate_content(new_context)
                text = response.text.strip()
                # json_response = json.loads(text)
                # result = json_response[example_cases]
                gemini_responses.append(text)
            except:
                gemini_responses.append("ERROR")

            time.sleep(60)
            counter += 1
        else:
            gemini_responses.append("NOT SENT")
            
    return gemini_responses


def segment_image(image_path, text_prompt):
    """
    Segments an image and uses Gemini to verify if segments match the prompt.

    Args:
      image_path: Path to the image.
      text_prompt: Text prompt describing the object to segment.

    Returns:
      A list of masks, confidence scores, and Gemini's YES/NO responses.
    """
    # Generate masks using your SAM 2 setup
    [generated_masks, _scores] = calculate_sam_masks(image_path)

    # Calculate similarity between text embedding and mask embeddings
    clip_confidences = calculate_clips_similarity(
        image_path, generated_masks, text_prompt
    )

    # Calculate Gemini responses
    gemini_responses = calculate_gemini_response(
        image_path, generated_masks, text_prompt
    )

    return generated_masks, clip_confidences, gemini_responses


def plot_results(image_path, masks, confidences, gemini_responses):
    """
    Plots the image with the generated masks and confidences.

    Args:
      image_path: Path to the image.
      masks: A list of masks.
      confidences: A list of confidence scores.
      gemini_responses: A list of Gemini responses.
    """
    image = Image.open(image_path)
    num_masks = len(masks)
    num_cols = 4
    num_rows = (num_masks + num_cols - 1) // num_cols

    plt.figure(figsize=(15, 5 * num_rows))
    for i, (mask, confidence, gemini_response) in enumerate(
        zip(masks, confidences, gemini_responses)
    ):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5)
        plt.title(
            f"Mask {i+1}\n(Confidence: {confidence:.2f})\n(Gemini: {gemini_response})"
        )
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# Load the models
clip_model, preprocess = load_clip_model()
sam2, mask_generator = load_sam_model()
gemini_model, context, example_cases = load_gemini_model()

if __name__ == "__main__":
    # Example usage
    image_path = "./cars.jpg"
    text_prompt = "a pink car"
    masks, confidences, gemini_responses = segment_image(image_path, text_prompt)
    plot_results(image_path, masks, confidences, gemini_responses)
