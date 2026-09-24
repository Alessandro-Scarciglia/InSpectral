import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Remove frames containing paths belonging to specified folders."
    )

    parser.add_argument(
        "input_json",
        type=Path,
        help="Input transforms.json file",
    )

    parser.add_argument(
        "folders",
        nargs="+",
        help="Folder names to filter out, e.g. mask infrared depth",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: transforms_rgb.json)",
    )

    args = parser.parse_args()

    input_json = args.input_json
    output_json = args.output or input_json.parent / "transforms_rgb.json"

    if not input_json.exists():
        raise FileNotFoundError(f"Input file not found: {input_json}")

    # Normalize folder names
    folders = {folder.strip("/") for folder in args.folders}

    # Load JSON
    with input_json.open("r") as f:
        data = json.load(f)

    if "frames" not in data:
        raise ValueError("JSON does not contain a 'frames' field.")

    original_frames = data["frames"]
    filtered_frames = []

    removed = 0

    for frame in original_frames:
        remove_frame = False

        # Check every string value in the frame.
        for key, value in frame.items():
            if not isinstance(value, str):
                continue

            path_parts = Path(value).parts

            if any(folder in path_parts for folder in folders):
                remove_frame = True
                break

        if remove_frame:
            removed += 1
        else:
            filtered_frames.append(frame)

    data["frames"] = filtered_frames

    # Save result
    with output_json.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"Input:              {input_json}")
    print(f"Output:             {output_json}")
    print(f"Original frames:    {len(original_frames)}")
    print(f"Removed frames:     {removed}")
    print(f"Remaining frames:   {len(filtered_frames)}")
    print(f"Filtered folders:   {', '.join(sorted(folders))}")


if __name__ == "__main__":
    main()