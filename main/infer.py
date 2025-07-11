import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import os
import argparse


def get_filenames_from_dir(directory):
    """Get list of filenames from a directory"""
    filenames = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            filenames.append(filename)
    return filenames


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Infer")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Directory path for input images")
    parser.add_argument("--label_path", type=str, required=True,
                        help="Directory path for mask images")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Output directory path for generated images")
    parser.add_argument("--prompt", type=str, default="a photo of hta",
                        help="Text prompt for generation")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Random seed value")

    args = parser.parse_args()

    # Create output directory if not exists
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # Load model pipeline
    pipeline = AutoPipelineForInpainting.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    pipeline.enable_model_cpu_offload()

    # Get input file list
    input_list = get_filenames_from_dir(args.input_path)

    w = 0  # Input image counter
    # Process each mask in label directory
    for root, dirs, files in os.walk(args.label_path):
        for file in files:
            mask_path = os.path.join(args.label_path, file)
            input_img_path = os.path.join(args.input_path, input_list[w % len(input_list)])

            # Load input and mask images
            init_image = load_image(input_img_path)
            mask_image = load_image(mask_path)

            out_image_path = os.path.join(args.out_path, file)

            # Set up generator with seed
            generator = torch.Generator("cuda").manual_seed(args.seed)

            # Generate inpainting result
            result = pipeline(
                prompt=args.prompt,
                image=init_image,
                mask_image=mask_image,
                generator=generator
            )
            image = result.images[0]

            # Maintain original dimensions
            target_width, target_height = init_image.size
            image = image.resize((target_width, target_height))

            # Save output
            image.save(out_image_path)
            w += 1


if __name__ == "__main__":
    main()