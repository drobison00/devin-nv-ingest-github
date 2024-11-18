import os
import base64
import json
import traceback
import click
import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from PIL import Image
import io
import numpy as np
import easyocr
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import matplotlib.patches as patches

# Initialize Table Detection model and processor
detection_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
detection_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

# Initialize Table Structure Recognition model and processor
structure_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
structure_model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])


def process_image(base64_string, display_overlay=False):
    try:
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        print(f"[DEBUG] Image size: {image.size}, Mode: {image.mode}")

        # Prepare inputs for table detection
        inputs = detection_processor(images=image, return_tensors="pt")
        print(f"[DEBUG] Inputs prepared with shape: {inputs['pixel_values'].shape}")

        # Pass inputs through the detection model
        outputs = detection_model(**inputs)
        print(f"[DEBUG] Outputs received with logits shape: {outputs.logits.shape}")

        # Post-process outputs to get detected tables
        tables = detection_processor.post_process_object_detection(
            outputs, threshold=0.9, target_sizes=[image.size[::-1]])[0]
        print(f"[DEBUG] Detected {len(tables['scores'])} tables")

        if len(tables['scores']) == 0:
            print("[WARNING] No tables detected in the image.")
            return ""

        # For overlay, collect bounding boxes
        table_bboxes = []
        cell_bboxes = []

        # Extract tables and their contents
        table_texts = []
        for idx, (score, label, box) in enumerate(zip(tables['scores'], tables['labels'], tables['boxes'])):
            print(f"[DEBUG] Processing table {idx + 1} with confidence {score.item():.3f}")

            # Expand the bounding box by 10%
            box = box.tolist()
            width = box[2] - box[0]
            height = box[3] - box[1]
            margin_x = 0.1 * width
            margin_y = 0.1 * height
            expanded_box = [
                max(0, int(box[0] - margin_x)),  # Ensure the box stays within image bounds
                max(0, int(box[1] - margin_y)),
                min(image.size[0], int(box[2] + margin_x)),
                min(image.size[1], int(box[3] + margin_y))
            ]
            print(f"[DEBUG] Original box: {box}, Expanded box: {expanded_box}")
            table_bboxes.append(expanded_box)

            # Extract the table from the image
            table_image = image.crop(expanded_box)
            print(f"[DEBUG] Cropped table image size: {table_image.size}")

            # Prepare inputs for structure recognition
            structure_inputs = structure_processor(images=table_image, return_tensors="pt")
            print(f"[DEBUG] Structure inputs prepared with shape: {structure_inputs['pixel_values'].shape}")

            # Pass inputs through the structure model
            structure_outputs = structure_model(**structure_inputs)
            print(f"[DEBUG] Structure outputs received with logits shape: {structure_outputs.logits.shape}")

            # Use post-processing to get cells
            cells = get_cells(structure_outputs, table_image.size, structure_processor, structure_model)
            print(f"[DEBUG] Detected {len(cells)} cells")

            if len(cells) == 0:
                print("[WARNING] No table rows or columns detected in the cropped table image.")
                continue

            # Adjust cell bounding boxes to the original image coordinates
            for cell in cells:
                adjusted_bbox = [
                    cell['bbox'][0] + expanded_box[0],
                    cell['bbox'][1] + expanded_box[1],
                    cell['bbox'][2] + expanded_box[0],
                    cell['bbox'][3] + expanded_box[1]
                ]
                cell_bboxes.append({'bbox': adjusted_bbox, 'label': cell['label']})

            cell_coordinates = get_cell_coordinates_by_row(cells)
            print(f"[DEBUG] Number of rows detected: {len(cell_coordinates)}")

            if len(cell_coordinates) == 0:
                print("[WARNING] No cell coordinates could be computed.")
                continue

            data = apply_ocr(cell_coordinates, table_image)
            print(f"[DEBUG] OCR data for table {idx + 1}: {data}")

            table_texts.append(data)

        # Generate markdown output
        markdown_output = []
        for idx, table_data in enumerate(table_texts):
            markdown_output.append(f"#### Table {idx + 1}")
            for row_data in table_data.values():
                markdown_output.append("| " + " | ".join(row_data) + " |")
            markdown_output.append("\n")
        print(f"[DEBUG] Markdown output generated")

        # If display_overlay is True, display the image with overlays
        if display_overlay:
            print("[DEBUG] Displaying image with detection overlays")
            plt.figure(figsize=(12, 12))
            plt.imshow(image)
            ax = plt.gca()

            # Plot table bounding boxes
            for bbox in table_bboxes:
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(bbox[0], bbox[1] - 10, 'Table', color='r', fontsize=12, weight='bold')

            # Plot cell bounding boxes
            for cell in cell_bboxes:
                bbox = cell['bbox']
                label = cell['label']
                color = 'g' if label == 'table row' else 'b'
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                    linewidth=1,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)
                # Optional: Uncomment to display labels
                # ax.text(bbox[0], bbox[1] - 5, label, color=color, fontsize=8)

            plt.axis('off')
            plt.title("Image with Detection Overlays")
            plt.show()

        return "\n".join(markdown_output)

    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] An error occurred in process_image: {e}")
        raise


def get_cells(outputs, img_size, processor, model):
    """
    Extract table rows and columns from model outputs using the post-processing method.

    Parameters:
        outputs: Model outputs containing logits and bounding box predictions.
        img_size: Tuple of (width, height) of the image.
        processor: The image processor used for post-processing.
        model: The structure recognition model (needed for id2label).

    Returns:
        A list of detected cells (rows and columns) with their labels, scores, and bounding boxes.
    """
    try:
        # Use the processor's post-processing method
        results = processor.post_process_object_detection(
            outputs, threshold=0.3, target_sizes=[img_size[::-1]])[0]
        print(f"[DEBUG] Post-processed {len(results['scores'])} detections")

        cells = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            class_label = model.config.id2label[label.item()]
            if class_label in ['table row', 'table column']:
                bbox = [float(b) for b in box.tolist()]
                print(f"[DEBUG] Detected {class_label} with score {score.item():.3f} at bbox {bbox}")
                cells.append({
                    'label': class_label,
                    'score': score.item(),
                    'bbox': bbox
                })
            else:
                print(f"[DEBUG] Ignoring detected {class_label} with score {score.item():.3f}")
        print(f"[DEBUG] Total detected cells: {len(cells)}")
        return cells

    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] An error occurred in get_cells: {e}")
        raise


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def get_cell_coordinates_by_row(table_data):
    # Extract rows and columns
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    # Sort rows and columns
    rows.sort(key=lambda x: x['bbox'][1])  # Top to bottom
    columns.sort(key=lambda x: x['bbox'][0])  # Left to right

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    # Generate cell coordinates
    cell_coordinates = []
    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x['column'][0])

        # Append row information
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells})

    return cell_coordinates


def apply_ocr(cell_coordinates, cropped_table):
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(tqdm(cell_coordinates, desc="Processing rows")):
        row_text = []
        for cell in row["cells"]:
            # Crop cell out of image
            cell_bbox = cell["cell"]
            cell_image = cropped_table.crop(cell_bbox)
            # Apply OCR
            result = reader.readtext(np.array(cell_image))
            if len(result) > 0:
                text = " ".join([x[1] for x in result])
                row_text.append(text)
            else:
                row_text.append("")

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[idx] = row_text

    # Pad rows to have the same number of columns
    for row, row_data in data.items():
        if len(row_data) < max_num_columns:
            row_data.extend([""] * (max_num_columns - len(row_data)))
        data[row] = row_data

    return data


@click.command()
@click.argument('input_file', type=click.Path(exists=True), required=False)
@click.option('--huggingface', is_flag=True, help="Use an example image from Hugging Face Hub.")
@click.option('--hf-repo-id', type=str, default="nielsr/example-pdf", help="Hugging Face repo ID for an example image.")
@click.option('--hf-filename', type=str, default="image.png",
              help="Filename in the Hugging Face repo for the example image.")
@click.option('--display-overlay', is_flag=True, help="Display the image with table/cell detection overlay.")
def process_input(input_file, huggingface, hf_repo_id, hf_filename, display_overlay):
    """
    Process a JSON file containing base64-encoded PNG metadata, a raw image file, or a Hugging Face example image.

    INPUT_FILE: Path to the JSON file or raw image file to process. If not provided, use the --huggingface option.
    """
    try:
        if huggingface:
            # Process the example image from Hugging Face Hub
            click.echo(f"Downloading image from Hugging Face: {hf_repo_id}/{hf_filename}")
            file_path = hf_hub_download(repo_id=hf_repo_id, repo_type="dataset", filename=hf_filename)
            image = Image.open(file_path).convert("RGB")

            # Display the image
            display_image(image, title="Hugging Face Example Image")

            # Prepare the image as base64 to reuse the same processing logic
            base64_image = image_to_base64(image)

            markdown = process_image(base64_image, display_overlay=display_overlay)
            click.echo(f"### Results from Hugging Face example image:\n{markdown}\n")

        elif input_file:
            # Determine if input_file is a JSON file or a raw image
            if input_file.endswith('.json'):
                # Process the JSON file
                with open(input_file, "r") as file:
                    data = json.load(file)

                for idx, entry in enumerate(data):
                    if "metadata" in entry and "content" in entry["metadata"]:
                        try:
                            click.echo(f"Processing entry {idx} in {input_file}...")
                            markdown = process_image(entry["metadata"]["content"], display_overlay=display_overlay)
                            click.echo(f"### Results for entry {idx}:\n{markdown}\n")
                        except Exception as e:
                            click.echo(f"Error processing entry {idx}: {e}")
            else:
                # Assume it's a raw image file
                click.echo(f"Processing raw image file: {input_file}")
                image = Image.open(input_file).convert("RGB")

                # Display the image
                # display_image(image, title="Raw Image File")

                # Convert the image to base64
                base64_image = image_to_base64(image)

                markdown = process_image(base64_image, display_overlay=display_overlay)
                click.echo(f"### Results from raw image file:\n{markdown}\n")

        else:
            click.echo("Please provide an input file or use the --huggingface option.")

    except Exception as e:
        traceback.print_exc()
        click.echo(f"Error: {e}")


def display_image(image, title="Image"):
    """
    Display a PIL image using matplotlib.
    """
    width, height = image.size
    resized_image = image.resize((int(0.6 * width), int(0.6 * height)))
    plt.figure(figsize=(8, 8))
    plt.imshow(resized_image)
    plt.axis("off")
    plt.title(title)
    plt.show()


def image_to_base64(image):
    """
    Convert a PIL image to a base64-encoded string.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


if __name__ == "__main__":
    process_input()
