![2.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/kBMZ3tkdVCN8O0z-FkNuO.png)

# Flood-Image-Detection

> Flood-Image-Detection is a vision-language encoder model fine-tuned from `google/siglip2-base-patch16-512` for **binary image classification**. It is trained to detect whether an image contains a **flooded scene** or **non-flooded** environment. The model uses the `SiglipForImageClassification` architecture.

> [!note]
SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features :  https://arxiv.org/pdf/2502.14786

```py
Classification Report:
               precision    recall  f1-score   support

Flooded Scene     0.9172    0.9458    0.9313       609
  Non Flooded     0.9744    0.9603    0.9673      1309

     accuracy                         0.9557      1918
    macro avg     0.9458    0.9530    0.9493      1918
 weighted avg     0.9562    0.9557    0.9559      1918
 ```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/T-KTVwt2YWoEjg6cB_rgh.png)


---

## Label Space: 2 Classes

```
Class 0: Flooded Scene  
Class 1: Non Flooded
```

---

## Install Dependencies

```bash
pip install -q transformers torch pillow gradio hf_xet
```

---

## Inference Code

```python
import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/flood-image-detection"  # Update with actual model name on Hugging Face
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Updated label mapping
id2label = {
    "0": "Flooded Scene",
    "1": "Non Flooded"
}

def classify_image(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }

    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=2, label="Flood Detection"),
    title="Flood-Image-Detection",
    description="Upload an image to detect whether the scene is flooded or not."
)

if __name__ == "__main__":
    iface.launch()
```

---

## Intended Use

`Flood-Image-Detection` is designed for:

* **Disaster Monitoring** – Rapid detection of flood-affected areas from imagery.
* **Environmental Analysis** – Track flooding patterns across regions using image datasets.
* **Crisis Response** – Assist emergency services in identifying critical zones.
* **Surveillance and Safety** – Monitor infrastructure or locations for flood exposure.
* **Smart Alert Systems** – Integrate with IoT or camera feeds for automated flood alerts.
