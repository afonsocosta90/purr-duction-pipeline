# demo/demo_data — Sample Images

Two ready-to-use test images for the Streamlit demo — one of each class the
model was trained on:

| File | Expected label | Shown in UI as |
|------|----------------|----------------|
| `cat_sample_1.jpg` | `cat` | Cat |
| `not_cat_sample_1.jpg` | `not_cat` | Dog |

The model is a cat-vs-dog classifier (Oxford-IIIT Pet dataset), so `not_cat`
is a dog. These two samples are in-distribution and classify with high
confidence.

## Using with the demo

1. Start the demo stack: `python demo/launch.py` (or `make demo`)
2. Open `http://localhost:8501`
3. In the **Live Prediction** tab, keep **🖼️ Sample images** selected and
   click **Cat** or **Dog** to classify it
4. Observe the real-time prediction card and confidence gauge
5. Submit feedback using the radio buttons below the result

## Adding your own images

Switch the input mode to **📤 Upload your own** and drag in (or browse for)
any `.jpg`, `.jpeg`, `.png`, or `.webp` image. File size limit: 10 MB.

Note: anything that is neither a cat nor a dog is out-of-distribution and may
be misclassified — even confidently.

## Source

Images are sampled from the Oxford-IIIT Pet Dataset (a cat class and a dog
class).
