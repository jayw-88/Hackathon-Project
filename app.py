import cv2
import numpy as np
import gradio as gr
from inference_sdk import InferenceHTTPClient
from groq import Groq
import os
from gradio.themes.base import Base
from gradio.themes.utils import colors

custom_theme = Base(
    primary_hue=colors.blue,
    secondary_hue=colors.zinc,
    neutral_hue=colors.zinc
).set(
    body_background_fill="rgba(255,255,255,0.6)",
    button_primary_background_fill="#3b82f6",
    shadow_drop="drop-shadow(0px 4px 6px rgba(0, 0, 0, 0.1))"
)

# Initialize Groq AI API key
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="96JPdCoiFFBc91M91y9i"
)

MODEL_ID = "fossil-scanner-v1-hs3pw/2"

def draw_fixed_label(img, label, confidence):
    label_text = label.title()
    conf_text = f"{confidence*100:.2f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_scale, conf_scale = 0.9, 0.7
    label_thick, conf_thick = 2, 1
    (lw, lh), _ = cv2.getTextSize(label_text, font, label_scale, label_thick)
    (cw, ch), _ = cv2.getTextSize(conf_text, font, conf_scale, conf_thick)
    pad = 12
    x, y = 35, 65
    box_w = max(lw, cw) + pad * 2
    box_h = lh + ch + pad * 3

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y - lh - pad), (x + box_w, y + box_h - lh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    cv2.putText(img, label_text, (x + pad, y),
                font, label_scale, (255, 255, 255), label_thick, cv2.LINE_AA)
    cv2.putText(img, conf_text, (x + pad, y + lh + pad),
                font, conf_scale, (144, 238, 144), conf_thick, cv2.LINE_AA)
    cv2.rectangle(img, (x + pad, y + lh + pad + 5),
                  (x + pad + int(confidence * (box_w - pad * 2)), y + lh + pad + 8),
                  (0, 255, 0), -1)
    return img

def process_image(frame):
    if frame is None:
        return None, "No image uploaded."
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite("temp.jpg", img)

    try:
        result = CLIENT.infer("temp.jpg", model_id=MODEL_ID)
    except Exception as e:
        return None, f"Error: {e}"

    predictions = result.get("predictions", [])
    if not predictions:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "No fossils detected."

    overlay = img.copy()

    if "x" in predictions[0] and "y" in predictions[0]:
        for pred in predictions:
            x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
            x2, y2 = min(img.shape[1], x + w // 2), min(img.shape[0], y + h // 2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 5000:
                cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 3)

    merged = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)
    label, confidence = predictions[0]["class"], predictions[0]["confidence"]
    user = f"Give a thorough description on {label}. Put it in the format: a general one-paragraph description, then a description of physical characteristics and composition, followed by a list of uses and significance of the artifact. (Don't include sources)"

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": user}],
            model="llama-3.1-8b-instant"
        )
        response = chat_completion.choices[0].message.content
    except Exception as e:
        response = f"Failed to generate description: {e}"

    final_img = draw_fixed_label(merged.copy(), label, confidence)
    output = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    info = f"**{label.title()}** - {confidence*100:.2f}% Confidence\n\n**Fossil Description:**\n{response}"
    return output, info

app = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Fossil Image"),
    outputs=[
        gr.Image(type="numpy", label="Result"),
        gr.Markdown(label="Prediction Details")
    ],
    title="AI Fossil Scanner",
    description="To input a fossil image, one may upload it, paste it, or use a webcam. AI will detect the fossil and show how confident it is. A detailed description of the fossil will also be provided.",
    theme=custom_theme
)

if __name__ == "__main__":
    app.launch(share = True)
