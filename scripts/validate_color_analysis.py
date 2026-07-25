import sys
import json
import cv2
import math
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


CONFIG_PATH = (
    PROJECT_ROOT
    / "ai"
    / "configs"
    / "phase3_config.yaml"
)

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


COLOR_METADATA_PATH = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "color_analysis_object_aware"
    / "color_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "color_validation_object_aware"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


with open(COLOR_METADATA_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)


items = []


for track in data["tracks"].values():

    crop_results = track.get("crop_color_results", [])

    if not crop_results:
        continue

    crop_path = Path(crop_results[0]["crop_path"])

    if not crop_path.exists():
        continue

    track_id = track["track_id"]
    class_name = track["class_name"]

    final_analysis = track["final_color_analysis"]


    if class_name in {"pedestrian", "people"}:

        upper = final_analysis[
            "upper_body"
        ]["primary_color"]

        lower = final_analysis[
            "lower_body"
        ]["primary_color"]

        label = (
            f"ID {track_id} | {class_name} | "
            f"upper:{upper} lower:{lower}"
        )

    else:

        object_result = final_analysis[
            "object_color"
        ]

        primary = object_result[
            "primary_color"
        ]

        secondary = object_result[
            "secondary_color"
        ]

        confidence = object_result[
            "confidence"
        ]

        label = (
            f"ID {track_id} | {class_name} | "
            f"{primary}/{secondary} "
            f"({confidence:.2f})"
        )


    items.append(
        {
            "crop_path": crop_path,
            "label": label,
        }
    )


THUMBNAIL_WIDTH = 340
THUMBNAIL_HEIGHT = 280
LABEL_HEIGHT = 70
COLUMNS = 4


rows = math.ceil(len(items) / COLUMNS)


canvas = np.full(
    (
        rows * (THUMBNAIL_HEIGHT + LABEL_HEIGHT),
        COLUMNS * THUMBNAIL_WIDTH,
        3,
    ),
    255,
    dtype=np.uint8,
)


for index, item in enumerate(items):

    image = cv2.imread(str(item["crop_path"]))

    if image is None:
        continue

    height, width = image.shape[:2]

    scale = min(
        THUMBNAIL_WIDTH / width,
        THUMBNAIL_HEIGHT / height,
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    column = index % COLUMNS
    row = index // COLUMNS

    x_start = column * THUMBNAIL_WIDTH

    y_start = (
        row
        * (
            THUMBNAIL_HEIGHT
            + LABEL_HEIGHT
        )
    )

    image_x = (
        x_start
        + (THUMBNAIL_WIDTH - new_width) // 2
    )

    image_y = (
        y_start
        + (THUMBNAIL_HEIGHT - new_height) // 2
    )

    canvas[
        image_y:image_y + new_height,
        image_x:image_x + new_width,
    ] = resized


    cv2.putText(
        canvas,
        item["label"],
        (
            x_start + 5,
            y_start
            + THUMBNAIL_HEIGHT
            + 30,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


OUTPUT_PATH = (
    OUTPUT_DIR
    / "object_aware_color_validation.jpg"
)


success = cv2.imwrite(
    str(OUTPUT_PATH),
    canvas,
)


print("\n" + "=" * 70)
print("OBJECT-AWARE COLOUR VALIDATION CREATED")
print("=" * 70)

print("Tracks displayed :", len(items))
print("Image saved      :", success)
print("Output image     :", OUTPUT_PATH)

print("=" * 70)