import os
import time
import numpy as np
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw
from pathlib import Path

from .ocr import OCR
from . import LayoutRecognizer, TableStructureRecognizer

class VietDocEngine:
    """
    Main engine for Vietnamese document processing.
    Combines Layout Analysis, Table Structure Recognition, and OCR.
    """
    def __init__(self, threshold=0.5, max_workers=4, use_paddleocr_rec=True):
        """
        Initialize VietDocEngine

        Args:
            threshold: Detection threshold for layout recognizer
            max_workers: Number of parallel workers for table processing
            use_paddleocr_rec: If True, use PaddleOCR recognition (recommended, 85-90% accuracy)
                              If False, use VietOCR recognition (legacy, 75-80% accuracy)
        """
        self.threshold = threshold
        self.max_workers = max_workers  # Number of parallel workers for table processing
        self.layout_recognizer = LayoutRecognizer("layout")
        self.ocr = OCR(use_paddleocr_rec=use_paddleocr_rec)  # Pass recognizer flag
        self.table_recognizer = TableStructureRecognizer()

    def process_image(self, img, img_name="page", figure_save_dir="figures", image_link_prefix="figures", debug_path=None):
        """
        Process a single PIL Image and return concatenated markdown.
        Now handles full document reconstruction (Title, Text, Figures, Tables).
        
        Args:
            img: PIL Image
            img_name: Base name for saving figures
            figure_save_dir: Directory to save extracted figures (absolute or relative to CWD)
            image_link_prefix: Prefix for image links in Markdown (relative to Markdown file)
            debug_path: Path to save the annotated debug image (optional)
        """
        start_time = time.time()

        # Ensure figure directory exists
        os.makedirs(figure_save_dir, exist_ok=True)
        
        # Prepare visualization if debug_path is provided
        vis_draw = None
        vis_img = None
        if debug_path:
            vis_img = img.copy()
            vis_draw = ImageDraw.Draw(vis_img)

        # 1. Layout Recognition
        layouts = self.layout_recognizer.forward([img], thr=float(self.threshold))[0]
        region_and_pos = []

        # Create a mask for detected regions to find "leftover" areas
        # 0 = Unprocessed, 1 = Processed
        mask = Image.new("1", img.size, 0)
        draw = ImageDraw.Draw(mask)

        # Collect regions by type for parallel/batch processing
        text_regions_batch = []  # [(bbox, label, y_pos, region_index), ...]
        table_regions_batch = []  # [(region, y_pos, region_index), ...]

        for i, region in enumerate(layouts):
            bbox = self._get_bbox(region)
            label = region.get("type", "").lower()
            score = region.get("score", 1.0)
            y_pos = bbox[1]

            if score < self.threshold:
                continue
            
            # Draw debug info
            if vis_draw:
                color = "red" if label in ["table", "figure", "equation"] else "blue"
                vis_draw.rectangle(bbox, outline=color, width=3)
                vis_draw.text((bbox[0], bbox[1]), f"{label} ({score:.2f})", fill=color)

            # Mark this region as processed
            draw.rectangle(bbox, fill=1)

            if label == "table":
                # Collect for parallel processing
                table_regions_batch.append((region, y_pos, i))

            elif label == "figure":
                # Save figure image
                fig_filename = f"{img_name}_fig_{i}.jpg"
                fig_path = os.path.join(figure_save_dir, fig_filename)

                # Path for Markdown link
                md_link_path = f"{image_link_prefix}/{fig_filename}"
                if image_link_prefix == "":
                     md_link_path = fig_filename

                try:
                    crop_img = img.crop(bbox)
                    crop_img.save(fig_path)
                    # Store bbox for smart sorting (format: [x0, y0, x1, y1])
                    region_and_pos.append((y_pos, f"![Figure]({md_link_path})", bbox))
                except Exception as e:
                    logging.error(f"Failed to save figure {fig_path}: {e}")
                
                # FIX: Also run OCR on the figure region in case it contains text (misclassified)
                text_regions_batch.append((bbox, "figure_text", y_pos + 1, i)) # Add slight offset to y_pos

            else:
                # Text-based regions: Title, Text, Header, Footer, Captions, Equation
                # Collect for batch processing
                text_regions_batch.append((bbox, label, y_pos, i))

        # Parallel table processing
        if table_regions_batch:
            try:
                table_results = self._process_tables_parallel(img, table_regions_batch)
                for y_pos, markdown in table_results:
                    if markdown and markdown.strip():
                        region_and_pos.append((y_pos, markdown))
            except Exception as e:
                logging.error(f"Error in parallel table processing: {e}")
                # Fallback to sequential processing
                for region, y_pos, idx in table_regions_batch:
                    try:
                        markdown = self.extract_table_markdown(img, region)
                        if markdown.strip():
                            region_and_pos.append((y_pos, markdown))
                    except Exception as e2:
                        logging.error(f"Error extracting table {idx}: {e2}")

        # Batch OCR processing for all text regions
        if text_regions_batch:
            try:
                # Crop all images at once
                cropped_images = [np.array(img.crop(bbox)) for bbox, _, _, _ in text_regions_batch]

                # Batch OCR call
                batch_ocr_results = self._batch_ocr(cropped_images)

                # Process results
                for (bbox, label, y_pos, idx), ocr_results in zip(text_regions_batch, batch_ocr_results):
                    text_content = "\n".join([t[0] for _, t in ocr_results if t and t[0]])

                    if not text_content.strip():
                        continue

                    # Apply Formatting
                    if label == "title":
                        text_content = f"# {text_content}"
                    elif label in ["header", "footer"]:
                        text_content = f"_{text_content}_"
                    elif label in ["figure caption", "table caption"]:
                        text_content = f"*{text_content}*"
                    elif label == "equation":
                        text_content = f"$$ {text_content} $$"

                    # Store bbox for smart sorting (format: [x0, y0, x1, y1])
                    region_and_pos.append((y_pos, text_content, bbox))
            except Exception as e:
                logging.error(f"Error in batch OCR processing: {e}")
                # Fallback to individual processing if batch fails
                for bbox, label, y_pos, idx in text_regions_batch:
                    try:
                        crop_img = img.crop(bbox)
                        ocr_results = self.ocr(np.array(crop_img))
                        text_content = "\n".join([t[0] for _, t in ocr_results if t and t[0]])

                        if not text_content.strip():
                            continue

                        if label == "title":
                            text_content = f"# {text_content}"
                        elif label in ["header", "footer"]:
                            text_content = f"_{text_content}_"
                        elif label in ["figure caption", "table caption"]:
                            text_content = f"*{text_content}*"
                        elif label == "equation":
                            text_content = f"$$ {text_content} $$"

                        # Store bbox for smart sorting
                        region_and_pos.append((y_pos, text_content, bbox))
                    except Exception as e2:
                        logging.error(f"Error processing region {label}: {e2}")

        # 3. OCR remaining undetected areas (Leftovers) - with smart skip
        # We need to mask out the already processed parts to avoid duplicates
        inv_mask = mask.point(lambda p: 1 - p) # 1 = Keep, 0 = Hide

        if inv_mask.getbbox():
            # Calculate leftover area percentage
            lx0, ly0, lx1, ly1 = inv_mask.getbbox()
            leftover_area = (lx1 - lx0) * (ly1 - ly0)
            total_area = img.size[0] * img.size[1]
            leftover_ratio = leftover_area / total_area if total_area > 0 else 0

            # Skip if leftover is too small (< 3% of total area)
            if leftover_ratio < 0.03:
                logging.info(f"Skipping leftover OCR: only {leftover_ratio:.1%} of image area")
            else:
                # Create an image where processed regions are whited out
                white_bg = Image.new("RGB", img.size, (255, 255, 255))
                leftover_img = Image.composite(img, white_bg, inv_mask)

                # Crop to the bounding box of the remaining content to save OCR time
                leftover_crop = leftover_img.crop((lx0, ly0, lx1, ly1))

                # Additional check: skip if mostly white pixels
                leftover_array = np.array(leftover_crop)
                white_pixels = np.sum(np.all(leftover_array > 240, axis=-1))
                total_pixels = leftover_array.shape[0] * leftover_array.shape[1]
                white_ratio = white_pixels / total_pixels if total_pixels > 0 else 1.0

                # FIX: Increase threshold to 99% and FORCE OCR if no layout was detected
                # This ensures sparse legal documents (stamps, signatures) are not skipped
                force_ocr = len(layouts) == 0
                
                if white_ratio > 0.99 and not force_ocr:
                    logging.info(f"Skipping leftover OCR: {white_ratio:.1%} white pixels (Threshold: 99%)")
                else:
                    if force_ocr:
                        logging.info(f"Forcing leftover OCR because no layout was detected (White ratio: {white_ratio:.1%})")
                    
                    # Draw debug for leftover
                    if vis_draw:
                        vis_draw.rectangle((lx0, ly0, lx1, ly1), outline="green", width=2)
                        vis_draw.text((lx0, ly0), "leftover", fill="green")

                    try:
                        ocr_results = self.ocr(np.array(leftover_crop))
                        leftover_text = "\n".join([t[0] for _, t in ocr_results if t and t[0]])
                        if leftover_text.strip():
                            region_and_pos.append((ly0, leftover_text))
                    except Exception as e:
                        logging.error(f"Error processing leftovers: {e}")

        # Save debug image
        if debug_path and vis_img:
            try:
                vis_img.save(debug_path)
                logging.info(f"Saved debug visualization to: {debug_path}")
            except Exception as e:
                logging.error(f"Failed to save debug image: {e}")

        # 4. Sort regions with smart reading order (handles multi-column & same-line regions)
        region_and_pos = self._smart_sort_regions(region_and_pos)
        markdown_concat = "\n\n".join([item[1] for item in region_and_pos])

        elapsed = time.time() - start_time
        logging.info(f"Processed image in {elapsed:.2f} seconds")

        return markdown_concat

    def extract_table_markdown(self, img, table_region):
        bbox = self._get_bbox(table_region)
        table_img = img.crop(bbox)

        tb_cpns = self.table_recognizer([table_img])[0]
        boxes = self.ocr(np.array(table_img))

        # Sort and clean up boxes for table reconstruction
        boxes = LayoutRecognizer.sort_Y_firstly(
            [{"x0": b[0][0], "x1": b[1][0],
              "top": b[0][1], "text": t[0],
              "bottom": b[-1][1],
              "layout_type": "table",
              "page_number": 0} for b, t in boxes if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]],
            np.mean([b[-1][1] - b[0][1] for b, _ in boxes]) / 3 if boxes else 10
        )

        if not boxes:
            return ""

        def gather(kwd, fzy=10, ption=0.6):
            eles = LayoutRecognizer.sort_Y_firstly(
                [r for r in tb_cpns if re.match(kwd, r["label"])], fzy)
            eles = LayoutRecognizer.layouts_cleanup(boxes, eles, 5, ption)
            return LayoutRecognizer.sort_Y_firstly(eles, 0)

        headers = gather(r".*header$")
        rows = gather(r".* (row|header)")
        spans = gather(r".*spanning")
        clmns = sorted([r for r in tb_cpns if re.match(
            r"table column$", r["label"])], key=lambda x: x["x0"])
        clmns = LayoutRecognizer.layouts_cleanup(boxes, clmns, 5, 0.5)

        for b in boxes:
            self._map_cell_to_structure(b, rows, headers, clmns, spans)

        return TableStructureRecognizer.construct_table(boxes, markdown=True)

    def _process_tables_parallel(self, img, table_regions_batch):
        """
        Process multiple tables in parallel using ThreadPoolExecutor.

        Args:
            img: PIL Image
            table_regions_batch: List of (region, y_pos, region_index)

        Returns:
            List of (y_pos, markdown) tuples
        """
        def process_single_table(args):
            region, y_pos, idx = args
            try:
                markdown = self.extract_table_markdown(img, region)
                return (y_pos, markdown)
            except Exception as e:
                logging.error(f"Error processing table {idx}: {e}")
                return (y_pos, "")

        # Use ThreadPoolExecutor for parallel processing
        # Tables often involve I/O (OCR calls) so threads work well
        results = []

        # If only 1 table, no need for parallel processing
        if len(table_regions_batch) == 1:
            return [process_single_table(table_regions_batch[0])]

        # Parallel processing for multiple tables
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(table_regions_batch))) as executor:
            # Submit all tasks
            future_to_table = {
                executor.submit(process_single_table, table_data): table_data
                for table_data in table_regions_batch
            }

            # Collect results as they complete
            for future in as_completed(future_to_table):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    table_data = future_to_table[future]
                    logging.error(f"Table processing failed for region {table_data[2]}: {e}")

        return results

    def _batch_ocr(self, image_list):
        """
        Optimized batch OCR processing for multiple images.

        Strategy:
        1. Detect text boxes in all images (sequential, unavoidable)
        2. Collect ALL text boxes from all images
        3. Batch recognize ALL boxes at once (major speedup)
        4. Map results back to original images

        Returns list of OCR results, one per image.
        """
        if not image_list:
            return []

        device_id = 0
        all_boxes_info = []  # [(image_idx, box, img_crop), ...]
        results = [[] for _ in range(len(image_list))]

        # Phase 1: Detect boxes in all images
        for img_idx, img_array in enumerate(image_list):
            dt_boxes, _ = self.ocr.text_detector[device_id](img_array)

            if dt_boxes is None:
                continue

            dt_boxes = self.ocr.sorted_boxes(dt_boxes)

            # Crop all detected boxes
            for box in dt_boxes:
                img_crop = self.ocr.get_rotate_crop_image(img_array, box)
                all_boxes_info.append((img_idx, box, img_crop))

        # Phase 2: Batch recognize ALL boxes at once
        if all_boxes_info:
            all_crops = [crop for _, _, crop in all_boxes_info]
            rec_results, _ = self.ocr.text_recognizer[device_id](all_crops)

            # Phase 3: Map results back to original images
            for (img_idx, box, _), (text, score) in zip(all_boxes_info, rec_results):
                if score >= self.ocr.drop_score:
                    results[img_idx].append((box.tolist(), (text, score)))

        return results

    def _get_bbox(self, region):
        if "bbox" in region:
            return list(map(int, region["bbox"]))
        return list(map(int, [region.get("x0", 0), region.get("top", 0), region.get("x1", 0), region.get("bottom", 0)]))

    def _map_cell_to_structure(self, b, rows, headers, clmns, spans):
        ii = LayoutRecognizer.find_overlapped_with_threashold(b, rows, thr=0.3)
        if ii is not None:
            b["R"] = ii
            b["R_top"] = rows[ii]["top"]
            b["R_bott"] = rows[ii]["bottom"]

        ii = LayoutRecognizer.find_overlapped_with_threashold(b, headers, thr=0.3)
        if ii is not None:
            b["H_top"] = headers[ii]["top"]
            b["H_bott"] = headers[ii]["bottom"]
            b["H_left"] = headers[ii]["x0"]
            b["H_right"] = headers[ii]["x1"]
            b["H"] = ii

        ii = LayoutRecognizer.find_horizontally_tightest_fit(b, clmns)
        if ii is not None:
            b["C"] = ii
            b["C_left"] = clmns[ii]["x0"]
            b["C_right"] = clmns[ii]["x1"]

        ii = LayoutRecognizer.find_overlapped_with_threashold(b, spans, thr=0.3)
        if ii is not None:
            b["H_top"] = spans[ii]["top"]
            b["H_bott"] = spans[ii]["bottom"]
            b["H_left"] = spans[ii]["x0"]
            b["H_right"] = spans[ii]["x1"]
            b["SP"] = ii

    def _smart_sort_regions(self, region_and_pos, y_threshold=30):
        """
        Sort regions with smart reading order:
        - Top to bottom (primary)
        - Left to right for regions on same horizontal line (secondary)

        Args:
            region_and_pos: List of (y_pos, content) or (y_pos, content, bbox) tuples
            y_threshold: Pixels threshold to consider regions on same line (default: 30px)

        Returns:
            Sorted list maintaining same format
        """
        if not region_and_pos:
            return region_and_pos

        # Convert to dict format for sort_Y_firstly
        regions_dict = []
        for item in region_and_pos:
            y_pos = item[0]
            content = item[1]

            # Extract bbox if available (from line 97: region_and_pos.append((y_pos, markdown)))
            # Most regions don't have bbox stored, so we use x0=0 as fallback
            x0 = 0
            if len(item) > 2 and isinstance(item[2], (list, tuple)):
                bbox = item[2]
                x0 = bbox[0] if len(bbox) > 0 else 0

            regions_dict.append({
                "top": y_pos,
                "x0": x0,
                "content": content,
                "original": item  # Keep original for reconstruction
            })

        # Use existing sort_Y_firstly method (already handles Y + X sorting)
        sorted_regions = LayoutRecognizer.sort_Y_firstly(regions_dict, y_threshold)

        # Reconstruct original format
        result = [r["original"] for r in sorted_regions]

        return result

if __name__ == "__main__":
    # Example usage
    import argparse
    from . import init_in_out

    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output_dir', default="./outputs")
    parser.add_argument('--threshold', default=0.5, type=float)
    args = parser.parse_args()

    engine = VietDocEngine(threshold=args.threshold)
    images, outputs = init_in_out(args)

    for idx, img in enumerate(images):
        md = engine.process_image(img)
        out_path = outputs[idx] + ".md"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Saved: {out_path}")
