import os
import sys
import time
import uuid
import json
import zipfile
import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, send_file, send_from_directory, render_template

# Ensure the FAST RETRIEVAL VL DETECTOR directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import the model
from model import FastRetrievalVLDetector

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Configuration for bulk uploads (support up to 1000+ high-resolution images)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GB max payload
app.config["MAX_FORM_PARTS"] = 10000  # Allow up to 10,000 form parts in multipart/form-data

# Directories
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
SAMPLES_DIR = BASE_DIR / "test_results"

# In-memory application store
# image_id -> { "id": str, "filename": str, "path": Path, "width": int, "height": int, "proposals": dict, "detections": list }
IMAGE_STORE = {}

# Predefined color palette for query labels
QUERY_COLORS = [
    "#00f0ff", # Neon Cyan
    "#10b981", # Emerald
    "#f59e0b", # Amber
    "#8b5cf6", # Purple
    "#f43f5e", # Rose
    "#06b6d4", # Ocean Blue
    "#ec4899", # Pink
    "#84cc16", # Lime
    "#3b82f6", # Royal Blue
    "#d946ef", # Fuchsia
]

# Initialize detector model lazily or on startup
detector = None

def get_detector():
    global detector
    if detector is None:
        print("[VisionLabel AI] Loading FastRetrievalVLDetector into memory...")
        detector = FastRetrievalVLDetector(confidence_threshold=0.10)
        print("[VisionLabel AI] FastRetrievalVLDetector initialized successfully!")
    return detector

def assign_color(query_idx):
    return QUERY_COLORS[query_idx % len(QUERY_COLORS)]

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    return response

# -------------------------------------------------------------
# Frontend Routes
# -------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR / "static"), "index.html")

# -------------------------------------------------------------
# API: Status & Health
# -------------------------------------------------------------

@app.route("/api/status", methods=["GET"])
def api_status():
    global detector
    model_loaded = detector is not None
    return jsonify({
        "status": "ready",
        "model_loaded": model_loaded,
        "model_name": "Fast Retrieval Vision-Language Detector",
        "backbone": "FasterRCNN-MobileNetV3 + CLIP-ViT-B/32",
        "images_count": len(IMAGE_STORE),
        "device": "CPU"
    })

@app.route("/api/warmup", methods=["POST"])
def api_warmup():
    start = time.perf_counter()
    det = get_detector()
    lat = (time.perf_counter() - start) * 1000
    return jsonify({
        "success": True,
        "message": "Model loaded into memory",
        "init_latency_ms": round(lat, 2)
    })

# -------------------------------------------------------------
# API: Sample Datasets
# -------------------------------------------------------------

SAMPLE_PRESETS = [
    {
        "filename": "dog_playing_input.jpg",
        "title": "Golden Retriever in Park",
        "queries": ["dog", "grass", "collar"],
        "category": "Pets & Nature"
    },
    {
        "filename": "extended_street_traffic_input.jpg",
        "title": "City Street Traffic",
        "queries": ["taxi", "car", "traffic light", "street sign", "person"],
        "category": "Urban & Vehicles"
    },
    {
        "filename": "extended_working_desk_input.jpg",
        "title": "Productive Desk Setup",
        "queries": ["headphones", "keyboard", "notebook", "desk"],
        "category": "Office & Objects"
    },
    {
        "filename": "extended_kitchen_scene_input.jpg",
        "title": "Modern Kitchen Interior",
        "queries": ["sink", "cabinet", "faucet", "plate", "bottle"],
        "category": "Home & Architecture"
    },
    {
        "filename": "extended_cozy_cat_input.jpg",
        "title": "Cozy Cat on Blanket",
        "queries": ["cat", "whiskers", "blanket", "eyes"],
        "category": "Pets & Nature"
    }
]

@app.route("/api/samples", methods=["GET"])
def api_samples():
    available_samples = []
    for item in SAMPLE_PRESETS:
        file_path = SAMPLES_DIR / item["filename"]
        if file_path.exists():
            try:
                with Image.open(file_path) as img:
                    w, h = img.size
                available_samples.append({
                    "id": f"sample_{item['filename']}",
                    "filename": item["filename"],
                    "title": item["title"],
                    "queries": item["queries"],
                    "category": item["category"],
                    "width": w,
                    "height": h,
                    "url": f"/api/sample_image/{item['filename']}"
                })
            except Exception as e:
                print(f"Error reading sample {file_path}: {e}")
    return jsonify({"samples": available_samples})

@app.route("/api/sample_image/<filename>", methods=["GET"])
def api_get_sample_image(filename):
    file_path = SAMPLES_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "Sample file not found"}), 404
    return send_file(str(file_path), mimetype="image/jpeg")

@app.route("/api/load_sample", methods=["POST"])
def api_load_sample():
    data = request.json or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "Filename required"}), 400
    
    file_path = SAMPLES_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "Sample file not found"}), 404

    image_id = str(uuid.uuid4())[:8]
    with Image.open(file_path) as img:
        w, h = img.size
    
    # Save a reference in store
    IMAGE_STORE[image_id] = {
        "id": image_id,
        "filename": filename,
        "path": file_path,
        "width": w,
        "height": h,
        "proposals": None,
        "detections": []
    }

    return jsonify({
        "success": True,
        "image": {
            "id": image_id,
            "filename": filename,
            "width": w,
            "height": h,
            "url": f"/api/image/{image_id}",
            "status": "unprocessed",
            "detections": []
        }
    })

@app.route("/api/load_all_samples", methods=["POST"])
def api_load_all_samples():
    loaded_images = []
    for item in SAMPLE_PRESETS:
        file_path = SAMPLES_DIR / item["filename"]
        if file_path.exists():
            image_id = str(uuid.uuid4())[:8]
            with Image.open(file_path) as img:
                w, h = img.size
            
            IMAGE_STORE[image_id] = {
                "id": image_id,
                "filename": item["filename"],
                "path": file_path,
                "width": w,
                "height": h,
                "proposals": None,
                "detections": []
            }
            loaded_images.append({
                "id": image_id,
                "filename": item["filename"],
                "title": item["title"],
                "queries": item["queries"],
                "width": w,
                "height": h,
                "url": f"/api/image/{image_id}",
                "status": "unprocessed",
                "detections": []
            })
    return jsonify({"success": True, "images": loaded_images})

# -------------------------------------------------------------
# API: Image Upload & Serving
# -------------------------------------------------------------

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "files" not in request.files:
        return jsonify({"error": "No files attached in upload"}), 400
    
    uploaded_files = request.files.getlist("files")
    results = []

    for f in uploaded_files:
        if not f.filename:
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"]:
            continue
        
        image_id = str(uuid.uuid4())[:8]
        save_name = f"{image_id}_{f.filename}"
        save_path = UPLOAD_DIR / save_name
        f.save(save_path)

        try:
            with Image.open(save_path) as img:
                w, h = img.size
            
            IMAGE_STORE[image_id] = {
                "id": image_id,
                "filename": f.filename,
                "path": save_path,
                "width": w,
                "height": h,
                "proposals": None,
                "detections": []
            }
            results.append({
                "id": image_id,
                "filename": f.filename,
                "width": w,
                "height": h,
                "url": f"/api/image/{image_id}",
                "status": "unprocessed",
                "detections": []
            })
        except Exception as e:
            print(f"Error opening uploaded file {save_path}: {e}")

    return jsonify({"success": True, "images": results})

@app.route("/api/image/<image_id>", methods=["GET"])
def api_serve_image(image_id):
    if image_id not in IMAGE_STORE:
        return jsonify({"error": "Image not found"}), 404
    info = IMAGE_STORE[image_id]
    return send_file(str(info["path"]))

@app.route("/api/images", methods=["GET"])
def api_get_images():
    images_list = []
    for img_id, info in IMAGE_STORE.items():
        images_list.append({
            "id": img_id,
            "filename": info["filename"],
            "width": info["width"],
            "height": info["height"],
            "url": f"/api/image/{img_id}",
            "status": "annotated" if len(info["detections"]) > 0 else "unprocessed",
            "detections_count": len(info["detections"])
        })
    return jsonify({"images": images_list})

@app.route("/api/image/<image_id>", methods=["DELETE"])
def api_delete_image(image_id):
    if image_id in IMAGE_STORE:
        info = IMAGE_STORE.pop(image_id)
        # Attempt to delete file if it was uploaded
        if UPLOAD_DIR in info["path"].parents:
            try:
                os.remove(info["path"])
            except Exception:
                pass
        return jsonify({"success": True, "message": f"Image {image_id} removed"})
    return jsonify({"error": "Image not found"}), 404

@app.route("/api/clear", methods=["POST"])
def api_clear_images():
    IMAGE_STORE.clear()
    return jsonify({"success": True, "message": "All images cleared"})

# -------------------------------------------------------------
# API: Detection & Annotation Engine
# -------------------------------------------------------------

def run_detection_for_image(image_id, queries, similarity_threshold=0.20, iou_threshold=0.40):
    """
    Runs vision proposals (cached per image) and fast similarity matching for queries.
    """
    if image_id not in IMAGE_STORE:
        raise ValueError(f"Image ID {image_id} not found in store")
    
    info = IMAGE_STORE[image_id]
    det = get_detector()
    
    # 1. Extract image proposals if not already cached
    start_total = time.perf_counter()
    proposal_lat = 0.0
    
    if info["proposals"] is None:
        pil_img = Image.open(info["path"]).convert("RGB")
        proposal_data = det.extract_image_proposals(pil_img)
        info["proposals"] = proposal_data
        proposal_lat = proposal_data["latency_ms"]
    else:
        proposal_data = info["proposals"]
    
    # 2. Retrieve objects for each prompt query
    all_detections = []
    retrieval_lats = []
    
    for q_idx, query in enumerate(queries):
        query = query.strip()
        if not query:
            continue
        
        # Pre-cache or get cached prompt
        det.cache_prompt(query)
        
        # Fast retrieval
        raw_dets, ret_lat = det.retrieve_objects(
            proposal_data, 
            query, 
            similarity_threshold=similarity_threshold, 
            iou_threshold=iou_threshold
        )
        retrieval_lats.append(ret_lat)
        
        color = assign_color(q_idx)
        for d in raw_dets:
            all_detections.append({
                "id": str(uuid.uuid4())[:8],
                "query": query,
                "color": color,
                "box": d["box"], # [x1, y1, x2, y2]
                "similarity": d["similarity"],
                "objectness": d["objectness"],
                "combined_score": d["combined_score"]
            })

    total_lat = (time.perf_counter() - start_total) * 1000
    info["detections"] = all_detections
    
    return {
        "image_id": image_id,
        "detections": all_detections,
        "proposal_latency_ms": round(proposal_lat, 2),
        "retrieval_latency_ms": round(sum(retrieval_lats), 2),
        "total_latency_ms": round(total_lat, 2)
    }

@app.route("/api/detect", methods=["POST"])
def api_detect():
    data = request.json or {}
    image_id = data.get("image_id")
    queries = data.get("queries", [])
    sim_thresh = float(data.get("similarity_threshold", 0.20))
    iou_thresh = float(data.get("iou_threshold", 0.40))
    
    if not image_id or image_id not in IMAGE_STORE:
        return jsonify({"error": "Valid image_id is required"}), 400
    
    if not queries or len(queries) == 0:
        return jsonify({"error": "At least one query is required"}), 400

    try:
        res = run_detection_for_image(image_id, queries, sim_thresh, iou_thresh)
        return jsonify({"success": True, **res})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/detect_batch", methods=["POST"])
def api_detect_batch():
    """
    Annotates ALL or specified N images for given queries.
    """
    data = request.json or {}
    image_ids = data.get("image_ids")
    if not image_ids:
        # Default to all images in store
        image_ids = list(IMAGE_STORE.keys())
    
    queries = data.get("queries", [])
    sim_thresh = float(data.get("similarity_threshold", 0.20))
    iou_thresh = float(data.get("iou_threshold", 0.40))

    if not queries or len(queries) == 0:
        return jsonify({"error": "At least one query is required"}), 400

    batch_results = []
    total_objects = 0
    total_time = 0.0

    for img_id in image_ids:
        if img_id not in IMAGE_STORE:
            continue
        try:
            res = run_detection_for_image(img_id, queries, sim_thresh, iou_thresh)
            batch_results.append(res)
            total_objects += len(res["detections"])
            total_time += res["total_latency_ms"]
        except Exception as e:
            print(f"Error detecting for image {img_id}: {e}")
            batch_results.append({"image_id": img_id, "error": str(e), "detections": []})

    return jsonify({
        "success": True,
        "processed_count": len(batch_results),
        "total_detections": total_objects,
        "avg_latency_ms": round(total_time / max(len(batch_results), 1), 2),
        "results": batch_results
    })

# -------------------------------------------------------------
# API: Manual Box Editing (Add / Update / Delete)
# -------------------------------------------------------------

@app.route("/api/detections/<image_id>", methods=["PUT"])
def api_update_detections(image_id):
    """
    Allows user to update the detections for an image (e.g. after manual edits).
    """
    if image_id not in IMAGE_STORE:
        return jsonify({"error": "Image not found"}), 404
    data = request.json or {}
    detections = data.get("detections", [])
    IMAGE_STORE[image_id]["detections"] = detections
    return jsonify({"success": True, "detections": detections})

# -------------------------------------------------------------
# API: Export Formats (COCO, YOLO, Detailed JSON, Images ZIP)
# -------------------------------------------------------------

@app.route("/api/export/coco", methods=["GET", "POST"])
def api_export_coco():
    """
    Generates standard COCO 1.0 JSON format for all annotated images.
    """
    categories_map = {}
    cat_id_counter = 1
    
    coco_images = []
    coco_annotations = []
    coco_categories = []
    
    ann_id_counter = 1
    img_num_id = 1
    
    for img_id, info in IMAGE_STORE.items():
        w, h = info["width"], info["height"]
        current_img_num_id = img_num_id
        img_num_id += 1

        coco_images.append({
            "id": current_img_num_id,
            "file_name": info["filename"],
            "width": w,
            "height": h,
            "custom_id": img_id
        })

        for det in info.get("detections", []):
            q_label = det["query"]
            if q_label not in categories_map:
                categories_map[q_label] = cat_id_counter
                coco_categories.append({
                    "id": cat_id_counter,
                    "name": q_label,
                    "supercategory": "object"
                })
                cat_id_counter += 1
            
            c_id = categories_map[q_label]
            x1, y1, x2, y2 = det["box"]
            bw = round(x2 - x1, 2)
            bh = round(y2 - y1, 2)
            bx = round(x1, 2)
            by = round(y1, 2)
            area = round(bw * bh, 2)

            coco_annotations.append({
                "id": ann_id_counter,
                "image_id": current_img_num_id,
                "category_id": c_id,
                "bbox": [bx, by, bw, bh],
                "area": area,
                "iscrowd": 0,
                "score": round(det.get("similarity", 1.0), 3)
            })
            ann_id_counter += 1

    coco_dataset = {
        "info": {
            "description": "VisionLabel AI Auto-Annotated Dataset",
            "version": "1.0",
            "year": 2026,
            "contributor": "VisionLabel AI Studio",
            "date_created": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories
    }

    return jsonify(coco_dataset)

@app.route("/api/export/yolo", methods=["GET", "POST"])
def api_export_yolo():
    """
    Generates a ZIP file containing YOLO format txt annotations and classes.txt.
    """
    # 1. Determine all unique classes
    all_classes = []
    for info in IMAGE_STORE.values():
        for det in info.get("detections", []):
            q = det["query"]
            if q not in all_classes:
                all_classes.append(q)
    
    class_to_id = {cls_name: idx for idx, cls_name in enumerate(all_classes)}

    # 2. Package in-memory zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write classes.txt
        zf.writestr("classes.txt", "\n".join(all_classes))

        # Write each image's label file
        for img_id, info in IMAGE_STORE.items():
            base_stem = Path(info["filename"]).stem
            txt_lines = []
            img_w = float(info["width"])
            img_h = float(info["height"])

            for det in info.get("detections", []):
                cls_id = class_to_id[det["query"]]
                x1, y1, x2, y2 = det["box"]
                
                # Normalize coordinates for YOLO (center_x, center_y, width, height)
                cx = ((x1 + x2) / 2.0) / img_w
                cy = ((y1 + y2) / 2.0) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                txt_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            
            zf.writestr(f"labels/{base_stem}.txt", "\n".join(txt_lines))

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="yolo_annotations.zip"
    )

@app.route("/api/export/json", methods=["GET", "POST"])
def api_export_json():
    """
    Exports a structured JSON file of all images and annotations.
    """
    export_data = {
        "project": "VisionLabel AI Annotation Studio",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(IMAGE_STORE),
        "total_detections": sum(len(info.get("detections", [])) for info in IMAGE_STORE.values()),
        "images": [
            {
                "id": img_id,
                "filename": info["filename"],
                "width": info["width"],
                "height": info["height"],
                "detections": info.get("detections", [])
            }
            for img_id, info in IMAGE_STORE.items()
        ]
    }
    return jsonify(export_data)

@app.route("/api/export/annotated_images", methods=["GET", "POST"])
def api_export_annotated_images():
    """
    Renders bounding boxes onto every image and exports all annotated images in a ZIP.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_id, info in IMAGE_STORE.items():
            if not info["path"].exists():
                continue
            
            try:
                img = Image.open(info["path"]).convert("RGB")
                draw = ImageDraw.Draw(img)

                for det in info.get("detections", []):
                    x1, y1, x2, y2 = det["box"]
                    color = det.get("color", "#00f0ff")
                    query = det.get("query", "object")
                    sim = int(det.get("similarity", 0) * 100)
                    label = f"{query} {sim}%"

                    # Draw rectangle
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Draw label background
                    label_w = len(label) * 8 + 10
                    label_h = 18
                    draw.rectangle([x1, max(0, y1 - label_h), x1 + label_w, y1], fill=color)
                    draw.text((x1 + 4, max(0, y1 - label_h) + 2), label, fill="#000000")

                img_io = io.BytesIO()
                img.save(img_io, format="JPEG", quality=92)
                img_io.seek(0)
                
                base_stem = Path(info["filename"]).stem
                zf.writestr(f"annotated_{base_stem}.jpg", img_io.getvalue())
            except Exception as e:
                print(f"Error rendering annotated image {img_id}: {e}")

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="annotated_images.zip"
    )

if __name__ == "__main__":
    print("\n=======================================================")
    print("   VisionLabel AI - Fast Vision-Language Annotator")
    print("=======================================================")
    print("Starting server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
