import argparse
import datetime
import json
import os
import time

import gradio as gr
import requests

from lyra.conversation import (default_conversation, conv_templates,
                                        SeparatorStyle)
from lyra.constants import LOGDIR
from lyra.utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
import hashlib
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
import torch
import soundfile as sf
import numpy as np

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "Lyra Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}

vocoder = None

def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None, None, None) + (disable_btn,) * 5


def add_text(state, text, speech, image, video, image_process_mode, instruct_mode, request: gr.Request):
    print(state)
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    
    if len(text) <= 0 and image is None and speech is None and video is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None, None, None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None, None, None) + (
                no_change_btn,) * 5  
    
    print("***********input data print start***********")
    print("text", text)
    print("image", image)
    print("speech", speech)
    print("video", video)
    print("***********input data print end***********")

    if instruct_mode == 'speech':
        if video is not None:
            text = '<image>\n<speech>'
            text = (text, speech, image, video, image_process_mode)
            if state is None:
                state = default_conversation.copy()
            state.append_message(state.roles[0], text)
            state.append_message(state.roles[1], None)
            state.skip_next = False
            return (state, state.to_gradio_chatbot(), "", None, None, None) + (disable_btn,) * 5

        if image is not None:
            text = '<image>\n<speech>'
            text = (text, speech, image, None, image_process_mode)
            if state is None:
                state = default_conversation.copy()
            state.append_message(state.roles[0], text)
            state.append_message(state.roles[1], None)
            state.skip_next = False
            return (state, state.to_gradio_chatbot(), "", None, None, None) + (disable_btn,) * 5
    
        if speech is not None:
            text = ('<speech>', speech, None, None, image_process_mode)
            if state is None:
                state = default_conversation.copy()
            state.append_message(state.roles[0], text)
            state.append_message(state.roles[1], None)
            state.skip_next = False
            return (state, state.to_gradio_chatbot(), "", None, None, None) + (disable_btn,) * 5
        
        if state is None:
            state = default_conversation.copy()
        text = (text, None, None, None, image_process_mode)
        state.append_message(state.roles[0], text)
        state.append_message(state.roles[1], None)
        state.skip_next = False
        return (state, state.to_gradio_chatbot(), "", None, None, None) + (disable_btn,) * 5

        
        
    elif instruct_mode == 'text':
        if video is not None:
            text = '<image>\n' + text
            text = (text, speech, image, video, image_process_mode)
            if state is None:
                state = default_conversation.copy()
            state.append_message(state.roles[0], text)
            state.append_message(state.roles[1], None)
            state.skip_next = False
            return (state, state.to_gradio_chatbot(), "", None, None, None) + (disable_btn,) * 5
    
        if speech is not None:
            text = ('<speech>\n'+ text, speech, None, None, image_process_mode)
            if state is None:
                state = default_conversation.copy()
            state.append_message(state.roles[0], text)
            state.append_message(state.roles[1], None)
            state.skip_next = False
            return (state, state.to_gradio_chatbot(), "", None, None, None) + (disable_btn,) * 5

        
        if image is not None:
            text = '<image>\n' + text
            text = (text, speech, image, None, image_process_mode)
            if state is None:
                state = default_conversation.copy()
            state.append_message(state.roles[0], text)
            state.append_message(state.roles[1], None)
            state.skip_next = False
            return (state, state.to_gradio_chatbot(), "", None, None, None) + (disable_btn,) * 5
    

        if state is None:
            state = default_conversation.copy()
        text = (text, None, None, None, image_process_mode)
        state.append_message(state.roles[0], text)
        state.append_message(state.roles[1], None)
        state.skip_next = False
        return (state, state.to_gradio_chatbot(), "", None, None, None) + (disable_btn,) * 5


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot(), None, None, None) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        template_name = 'qwen2vl'

        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), "", disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return
    
    # Construct prompt
    state_tmp = state.copy()
    num_messages = len(state_tmp.messages)
    cur_prompt = state_tmp.messages[-2][1]
    if isinstance(cur_prompt, tuple):
        cur_prompt = cur_prompt[0]
    print(cur_prompt, len(state_tmp.messages))
    if '<speech>' in cur_prompt:
        state_tmp.messages = state_tmp.messages[-2:]
    else:
        for i in range(num_messages):
            if isinstance(state_tmp.messages[i][1], tuple):
                messages_tmp = list(state_tmp.messages[i][1])
                messages_tmp[0] = messages_tmp[0].replace('<speech>', '')
                state_tmp.messages[i][1] = tuple(messages_tmp)

    prompt = state_tmp.get_prompt()
    prompt = prompt.replace('<|im_start|>system\nYou are a helpful assistant.', '<|im_start|>system\nYou are a helpful multi-modality assistant called Lyra.')
    print(prompt)

    '''
    if '<speech>' in prompt.split("\n<|im_start|>user\n")[-1]:
        if '<image>' in prompt:
            prompt = "<|im_start|>system\nYou are a helpful multi-modality assistant called Lyra.<|im_end|>\n<|im_start|>user\n<image>\n<speech><|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = "<|im_start|>system\nYou are a helpful multi-modality assistant called Lyra.<|im_end|>\n<|im_start|>user\n<speech><|im_end|>\n<|im_start|>assistant\n"
    else:
        wo_speech_prompt = ""
        for i, term_prompt in enumerate(prompt.split("\n<|im_start|>user\n")):
            if '<speech>' not in term_prompt and (i != len(prompt.split("\n<|im_start|>user\n")) - 1):
                wo_speech_prompt += (term_prompt + "\n<|im_start|>user\n")
            elif '<speech>' in term_prompt and (i != len(prompt.split("\n<|im_start|>user\n")) - 1):
                if '<image>' in term_prompt:
                    wo_speech_prompt += (term_prompt.replace("<speech>", "") + "\n<|im_start|>user\n")
            if i == len(prompt.split("\n<|im_start|>user\n")) - 1:
                wo_speech_prompt += term_prompt
        prompt = wo_speech_prompt
    '''

    
    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep,
        "images": state.get_images(),
        "speeches": state.get_speeches(),
        "videos": state.get_videos(),
    }
    logger.info(f"==== request ====\n{pload}")

    pload['images'] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot(), None) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=30)
        num_generated_units = 0
        wav_list = []
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    if 'image' not in data.keys():
                        output = data["text"][len(prompt):].strip()
                        state.messages[-1][-1] = output + "▌"
                        
                    else:
                        output = (data["text"][len(prompt):].strip(), data["image"])
                        state.messages[-1][-1] = output
                    output_unit = list(map(int, data["unit"].strip().split()))
                    
                    new_units = output_unit[num_generated_units:]
                    if len(new_units) >= 40:
                        num_generated_units = len(output_unit)
                        x = {"code": torch.LongTensor(new_units).view(1, -1).cuda()}
                        wav = vocoder(x, True)
                        wav_list.append(wav.detach().cpu().numpy())

                    if len(wav_list) > 0:
                        wav_full = np.concatenate(wav_list)
                        return_value = (16000, wav_full)
                    else:
                        return_value = None
                        
                    yield (state, state.to_gradio_chatbot(), return_value) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot(), None) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), None) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return
    
    if num_generated_units < len(output_unit):
        new_units = output_unit[num_generated_units:]
        num_generated_units = len(output_unit)
        x = {
            "code": torch.LongTensor(new_units).view(1, -1).cuda()
        }
        wav = vocoder(x, True)
        wav_list.append(wav.detach().cpu().numpy())

    if len(wav_list) > 0:
        wav_full = np.concatenate(wav_list)
        return_value = (16000, wav_full)
    else:
        return_value = None

    if type(state.messages[-1][-1]) is not tuple:
        state.messages[-1][-1] = state.messages[-1][-1][:-1]
    ######################## wave output
    
    yield (state, state.to_gradio_chatbot(), return_value) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "images": state.get_images(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

lyra_svg_icon = """<svg xmlns="http://www.w3.org/2000/svg" version="1.1" xmlns:xlink="http://www.w3.org/1999/xlink" width="24" height="24" x="0" y="0" viewBox="0 0 464 464" style="enable-background:new 0 0 512 512" xml:space="preserve" class=""><g><path d="M370.016 269.328 194.672 93.984l-16.457 8.918 182.89 182.891zM300.688 248 288 235.312l-146.832 146.84c5.594 3.047 11.473 5.602 17.465 7.91zM141.207 122.945A147.277 147.277 0 0 0 64 252.575c0 12.288 1.68 24.167 4.52 35.593L204.688 152l-41.145-41.152zM341.055 322.793l12.097-22.336L312 259.313 175.832 395.48c11.426 2.84 23.305 4.52 35.594 4.52a147.277 147.277 0 0 0 129.629-77.207zM252.688 200 240 187.312 90.559 336.755c3.457 4.941 7.257 9.613 11.28 14.094zm0 0" fill="#50ace9" opacity="1" data-original="#000000" class=""></path><path d="M440 224h-12.945a23.98 23.98 0 0 0-21.102 12.566l-50.832 93.84A163.264 163.264 0 0 1 211.426 416C121.312 416 48 342.687 48 252.574c0-60.039 32.8-115.101 85.594-143.695l93.84-50.832A23.989 23.989 0 0 0 240 36.945V24c0-13.23-10.77-24-24-24s-24 10.77-24 24v12.77l-85.91 46.527A202.404 202.404 0 0 0 0 261.426C0 373.129 90.871 464 202.574 464a202.397 202.397 0 0 0 178.121-106.09L427.23 272H440c13.23 0 24-10.77 24-24s-10.77-24-24-24zM216 16c4.414 0 8 3.586 8 8s-3.586 8-8 8-8-3.586-8-8 3.586-8 8-8zM49.602 147.473l13.183 9.054a175.847 175.847 0 0 0-9.137 14.84l-14.023-7.71a190.266 190.266 0 0 1 9.977-16.184zM39.91 308.336l-15.277 4.762C18.903 294.68 16 275.473 16 256c0-26.207 5.184-51.617 15.426-75.512l14.71 6.297C36.755 208.68 32 231.97 32 256c0 17.855 2.664 35.465 7.91 52.336zM440 256c-4.414 0-8-3.586-8-8s3.586-8 8-8 8 3.586 8 8-3.586 8-8 8zm0 0" fill="#50ace9" opacity="1" data-original="#000000" class=""></path><path d="M228.688 176 216 163.312 73.945 305.368c2.313 6 4.864 11.871 7.91 17.465zM276.688 224 264 211.312l-150.848 150.84c4.48 4.024 9.153 7.825 14.094 11.282zm0 0" fill="#50ace9" opacity="1" data-original="#000000" class=""></path></g></svg>"""

title_markdown = (f"""
# {lyra_svg_icon}&nbsp;Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition
""")

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")


learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
""")

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""

def build_demo(embed_mode, vocoder, cur_dir=None, concurrency_count=10):
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="Lyra", theme='ParityError/Interstellar', css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)

                instruct_mode = gr.Radio(
                    ["speech", "text"],
                    value="speech",
                    label="Instruction Type")
                
                speech = gr.Audio(sources=["microphone", "upload"], label="Record your voice / Upload", type="filepath")
                
                imagebox = gr.Image(type="filepath", label='Upload image', height=300)
                videobox = gr.Video(label='Upload video', height=300)
                
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)
            

            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="Lyra Chatbot",
                    height=500,
                    layout="panel",
                )
                with gr.Row():
                    with gr.Column(scale=7):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Column(scale=20):
                    audio_output_box = gr.Audio(label="Lyra Speech Output", elem_classes="custom-audio")
                
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)
                with gr.Row():
                    if cur_dir is None:
                        cur_dir = os.path.dirname(os.path.abspath(__file__))
                    with gr.Column():
                        gr.Examples(examples=[
                            [f"examples/extreme_ironing.jpg", f"examples/extreme_ironing.mp3", 'speech'],
                            [f"examples/Chinese_painting.jpg", f"examples/Chinese_painting.mp3", 'speech'],
                        ], inputs=[imagebox, speech, instruct_mode])
                        gr.Examples(examples=[
                            [f"examples/pred_round1.wav", "Record the spoken words as text.", 'text'],
                        ], inputs=[speech, textbox, instruct_mode])
                    with gr.Column():
                        gr.Examples(examples=[
                            [f"examples/movement.mov", 'speech'],
                        ], inputs=[videobox, instruct_mode])
                        gr.Examples(examples=[
                            [f"examples/Hi.mp3", 'speech'],
                        ], inputs=[speech, instruct_mode])

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, clear_btn]
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        
        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, speech, imagebox, videobox] + btn_list,
            queue=False
        )
        
        textbox.submit(
            add_text,
            [state, textbox, speech, imagebox, videobox, image_process_mode, instruct_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot, audio_output_box] + btn_list,
            concurrency_limit=concurrency_count
        )

        submit_btn.click(
            add_text,
            [state, textbox, speech, imagebox, videobox, image_process_mode, instruct_mode],
            [state, chatbot, textbox, speech, imagebox, videobox] + btn_list
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot, audio_output_box] + btn_list,
            concurrency_limit=concurrency_count
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                _js=get_window_url_params
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo

def build_vocoder():
    global vocoder
    with open('model_zoo/audio/vocoder/config.json') as f:
        vocoder_cfg = json.load(f)
    vocoder = CodeHiFiGANVocoder('model_zoo/audio/vocoder/g_00500000', vocoder_cfg).cuda()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=16)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--ssl-certfile", type=str, default="lyra/serve/cert.pem")
    parser.add_argument("--ssl-keyfile", type=str, default="lyra/serve/key.pem")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    build_vocoder()
    
    logger.info(args)
    demo = build_demo(args.embed, vocoder, concurrency_count=args.concurrency_count)
    demo.queue(
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=True,
        show_error=True,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
        ssl_verify=False,
    )
