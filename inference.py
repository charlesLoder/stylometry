import torch
import argparse

import transformers
from transformers.image_utils import load_image
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import model_info
from PIL import Image
from typing import Optional
from yaspin import yaspin


DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

transformers.logging.set_verbosity_error()


def get_model_processor(model_id):
    """
    Load the processor for the given model ID. If it fails, try to load the base model's processor.
    Args:
        model_id (str): The model ID to load the processor for.
    Returns:
        processor (AutoProcessor): The loaded processor.
    """
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        return processor
    except Exception as e:
        if model_info(model_id).card_data.base_model:
            return get_model_processor(model_info(model_id).card_data.base_model)
        else:
            raise Exception(f"Failed to load processor for model {model_id}.") from e


def generate_ouput(
    model_id: str,
    prompt: str = "Describe this image",
    system_prompt: str = "You are a helpful assistant.",
    image: Optional[Image.Image] = None,
) -> str:
    processor = get_model_processor(model_id)

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)

    user_content = [{"type": "text", "text": prompt}]
    if image is not None:
        user_content.insert(0, {"type": "image", "image": image})
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": user_content},
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    full_text = generated_texts[0]

    # Remove the prompt from the output
    return full_text.split("Assistant:")[1].strip()


def main():
    parser = argparse.ArgumentParser(description="Generate output using a model.")
    parser.add_argument(
        "--model", type=str, required=True, help="Model ID to use for generation."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image",
        help="Prompt to use for generation.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt to use for generation.",
    )
    parser.add_argument(
        "--image", nargs="?", help="An image paths to use for generation."
    )

    args = parser.parse_args()

    image = None
    if args.image:
        image = load_image(args.image)

    spinner = yaspin()
    spinner.start()
    output = generate_ouput(args.model, args.prompt, args.system_prompt, image)
    spinner.stop()
    spinner.ok("🤖💬")
    print(output)


if __name__ == "__main__":
    main()
