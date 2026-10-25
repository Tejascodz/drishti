"""Write the Albumentations pipeline used for OCR fine-tuning.

Real crops get the same capture-chain corruptions the synthesiser applies, at a
lower rate, plus coarse dropout (partial occlusion) which the ICPR 2026 LRLPR
winners found important for robustness.

    python training/ocr/make_augmentation.py training/ocr/augment.yaml
"""

import sys

import albumentations as A


def build() -> A.Compose:
    return A.Compose(
        [
            A.Affine(scale=(0.92, 1.06), translate_percent=(-0.04, 0.04), rotate=(-4, 4), shear=(-6, 6), p=0.6),
            A.Perspective(scale=(0.02, 0.06), p=0.3),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=(3, 7)), A.Defocus(radius=(1, 2))], p=0.35),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            A.RandomGamma(gamma_limit=(70, 140), p=0.3),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=0.3),
            A.Downscale(scale_range=(0.35, 0.8), p=0.3),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
            A.ImageCompression(quality_range=(20, 90), p=0.4),
            A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(0.05, 0.25), hole_width_range=(0.03, 0.12), p=0.3),
        ]
    )


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "augment.yaml"
    A.save(build(), out, data_format="yaml")
    print(f"wrote {out}")
