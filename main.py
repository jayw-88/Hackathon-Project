import cv2
import gradio as gr
from inference_sdk import InferenceHTTPClient
from groq import Groq
import os
import streamlit as st
import time 
# Display Streamlit content
st.title("Streamlit App with Gradio Integration")

import subprocess
aaa = subprocess.Popen(["gradio", "gradio_interface.py"])

# Replace the Gradio interface URL with your generated share link
gradio_interface_url = "https://baa03635463a8706a5.gradio.live"

# Load the Gradio interface using an iframe
st.write(f'<iframe src="{gradio_interface_url}" width="800" height="600"></iframe>',
         unsafe_allow_html=True)

# Initialize Groq AI API key
client = Groq(api_key = os.environ["GROQ_API_KEY"])

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="96JPdCoiFFBc91M91y9i"
)

MODEL_ID = "fossil-scanner-v2-ncp2c-xfqbt/1"

def draw_fixed_label(img, label, confidence):
    label_text = label.title()
    conf_text = f"{confidence*100:.1f}%"
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
    user = f"Give a thorough description on {label} Put it in the format following a general one-paragraph description, then a description of physical charateristcs and composition. Then put a list of uses and signifance of the artifact. (Don't include sources)"
    chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": user}
                ],
                model="llama-3.1-8b-instant"
            )
    response = chat_completion.choices[0].message.content

    final_img = draw_fixed_label(merged.copy(), label, confidence)

    output = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    info = f"**{label.title()}** — {confidence*100:.2f}%\n**Description:**\n{response}"
    return output, info

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Upload a Fossil Image"),
    outputs=[
        gr.Image(type="numpy", label="Processed Result"),
        gr.Markdown(label="Prediction Details")
    ],
    title="AI Fossil Scanner",
    description="Upload a fossil image; AI will detect it automatically and give a detailed description on the given fossil. Bounding boxes will be outlined with a label card.",
    theme="default",
    examples=[["th.jpg"]]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
