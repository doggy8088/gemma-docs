# Inference

Render the segmentation masks:

```python
def visualize(image, masks):
    for mask, bbox in masks:
        mask = np.array(mask)
        binary_mask = (mask > 0.5).astype(np.uint8)


        # Place the binary mask onto the full mask
        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], binary_mask)
    return full_mask
```

That produces the blended output.
