# Fine-tuning

The helper below encodes an image for the API:

```python
def to_data_url(image_b64):
    return f"data:image/jpeg;base64,{image_b64}"
```

Running the cell produces:

![](assets/base64_image-01.png)

And a second output with alt text:

![png](assets/base64_image-02.png)
