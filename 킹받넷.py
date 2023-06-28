import gradio as gr
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler


def generate_image(image):
    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    caption = image_to_text(image)[0]["generated_text"]
    model_id = "CompVis/stable-diffusion-v1-4"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    model = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    model = model.to("cuda")
    generated_image = model(caption).images[0]
    return generated_image


inputs = gr.inputs.Image(source="webcam", type="numpy", label="Webcam")
outputs = gr.outputs.Image(type="numpy", label="Generated Image")


title = "<h1 style='color: #E69A8D'>킹받넷.com</h1>"
description = "<p style='color: #E69A8D; font-size: 18px'>웹캠을 켜서 사진을 찍고 <span style='color: #9400D3; font-weight: bold;'>재미있는 이미지</span>를 생성해보세요!</p>"
theme = "Edge"
allow_flagging = True


app = gr.Interface(
    fn=generate_image,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    theme=theme,
    allow_flagging=allow_flagging
)


if __name__ == "__main__":
    app.launch()

