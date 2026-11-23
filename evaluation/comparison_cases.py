from pathlib import Path


COMPARISON_ROOT = Path(
    "evaluation/comparisons"
)


COMPARISON_CASES = {

    "segmentation_test": {
        "description": (
            "Testing w and w/o segmentation embeddings. Testing 0/4/8/16 lengths."
        ),
        "experiments": [
            {
                "name": "Label Camera Poses",
                "folder": "results/mask_estimate",
            },
            {
                "name": "Noisy Camera Poses",
                "folder": "results/folder_2026-11-06_10-11-48",
            },
        ],
        "output": COMPARISON_ROOT / "results_pose_uncert",
    },


}