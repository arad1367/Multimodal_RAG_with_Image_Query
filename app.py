# import spaces --> just for Hugging Face deployment with ZeroGPU
import os
import gradio as gr
from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import torchvision
import subprocess

# Only For Hugging Face Deployment
# def install_poppler():
#     try:
#         subprocess.run(["pdfinfo"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     except FileNotFoundError:
#         print("Poppler not found. Installing...")
#         subprocess.run("apt-get update", shell=True)
#         subprocess.run("apt-get install -y poppler-utils", shell=True)

# install_poppler()

subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
                                                        trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)

# @spaces.GPU() <-- For HF and use ZeroGPU
def process_pdf_and_query(pdf_file, user_query):
    images = convert_from_path(pdf_file.name)
    num_images = len(images)

    RAG.index(
        input_path=pdf_file.name,
        index_name="image_index",
        store_collection_with_index=False,
        overwrite=True
    )

    results = RAG.search(user_query, k=1)
    if not results:
        return "No results found.", num_images

    image_index = results[0]["page_num"] - 1
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": images[image_index],
                },
                {"type": "text", "text": user_query},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0], num_images

css = """
body {
    font-family: Arial, sans-serif;
    background-color: #2b2b2b;
    color: #e0e0e0;
}
.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: #363636;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0,0,0,0.3);
}
.title {
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
    color: #50fa7b;
}
.submit-btn {
    background-color: #50fa7b;
    color: #282a36;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
    font-weight: bold;
}
.submit-btn:hover {
    background-color: #45c967;
}
.duplicate-button {
    background-color: #8be9fd;
    color: #282a36;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
    font-weight: bold;
    margin-top: 20px;
}
.duplicate-button:hover {
    background-color: #79c7d8;
}
a {
    color: #8be9fd;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
"""

explanation = """
<div style="background-color: #44475a; padding: 15px; border-radius: 5px; margin-bottom: 20px; color: #f8f8f2;">
    <h3 style="color: #50fa7b;">About Multimodal RAG</h3>
    <p>Multimodal RAG (Retrieval-Augmented Generation) combines text and image processing to provide more context-aware responses. This demo uses:</p>
    <ul>
        <li><strong style="color: #ffb86c;">ColPali</strong>: A multimodal retriever for efficient information retrieval from images and text.</li>
        <li><strong style="color: #ffb86c;">Byaldi</strong>: A new library by answer.ai that simplifies the use of ColPali.</li>
        <li><strong style="color: #ffb86c;">Qwen/Qwen2-VL-2B-Instruct</strong>: A large language model capable of processing both text and visual inputs.</li>
    </ul>
    <p>This combination allows for more accurate and context-aware responses to queries about uploaded PDFs.</p>
</div>
"""

footer = """
<div style="text-align: center; margin-top: 20px; color: #f8f8f2;">
    <a href="https://www.linkedin.com/in/pejman-ebrahimi-4a60151a7/" target="_blank">LinkedIn</a> |
    <a href="https://github.com/arad1367" target="_blank">GitHub</a> |
    <a href="https://arad1367.pythonanywhere.com/" target="_blank">Live demo of my PhD defense</a> |
    <a href="https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct" target="_blank">Qwen/Qwen2-VL-2B-Instruct</a> |
    <a href="https://github.com/AnswerDotAI/byaldi" target="_blank">Byaldi</a> |
    <a href="https://github.com/illuin-tech/colpali" target="_blank">ColPali</a>
    <br>
    Made with 💖 by Pejman Ebrahimi
</div>
"""

with gr.Blocks(css=css, theme='freddyaboulton/dracula_revamped') as demo:
    gr.HTML('<h1 style="text-align: center; font-size: 32px;"><a href="https://github.com/arad1367" target="_blank" style="text-decoration: none; color: #50fa7b;">Multimodal RAG with Image Query - By Pejman Ebrahimi</a></h1>')
    gr.HTML(explanation)
    pdf_input = gr.File(label="Upload PDF")
    query_input = gr.Textbox(label="Enter your query", placeholder="Ask a question about the PDF")
    submit_btn = gr.Button("Submit", elem_classes="submit-btn")
    output_text = gr.Textbox(label="Model Answer")
    output_images = gr.Textbox(label="Number of Images in PDF")
    
    submit_btn.click(process_pdf_and_query, inputs=[pdf_input, query_input], outputs=[output_text, output_images])
    
    gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")
    gr.HTML(footer)

demo.launch(debug=True)