# demo/demo_data — Sample Images

Five ready-to-use test images for the Streamlit demo:

| File | Expected label |
|------|---------------|
| `cat_sample_1.jpg` | `cat` |
| `cat_sample_2.jpg` | `cat` |
| `cat_sample_3.jpg` | `cat` |
| `not_cat_sample_1.jpg` | `not_cat` |
| `not_cat_sample_2.jpg` | `not_cat` |

## Using with the demo

1. Start the demo stack: `make demo`
2. Open `http://localhost:8501`
3. In the **Live Prediction** tab, drag any image above onto the uploader
4. Observe the real-time prediction card and confidence gauge
5. Submit feedback using the radio buttons below the result

## Adding your own images

Drop any `.jpg`, `.jpeg`, `.png`, or `.webp` image into this folder and
upload it via the Streamlit UI.  File size limit: 10 MB.

## Source

Images are sampled from the Oxford-IIIT Pet Dataset (cat classes)
and the not_cat subset of the training data.
