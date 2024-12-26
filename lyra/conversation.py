import dataclasses
from enum import auto, Enum
from typing import List, Tuple
import base64
from io import BytesIO
from PIL import Image


def img_to_base64(img_file_path):
    with open(img_file_path, "rb") as wav_file:
        wav_data = wav_file.read()
        base64_encoded = base64.b64encode(wav_data).decode('utf-8')
        return base64_encoded

def wav_to_base64(wav_file_path):
    with open(wav_file_path, "rb") as wav_file:
        wav_data = wav_file.read()
        base64_encoded = base64.b64encode(wav_data).decode('utf-8')
        return base64_encoded
    
def mov_to_base64(mov_file_path):
    with open(mov_file_path, "rb") as mov_file:
        mov_data = mov_file.read()
        base64_encoded = base64.b64encode(mov_data).decode('utf-8')
        return base64_encoded


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    SPEECH_PLAIN = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()
    LLAMA_3_1 = auto()
    QWEN2 = auto()
    QWEN2VL = auto()
    GEMMA = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            prompt_token =  messages[0][1][0]
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace(prompt_token, "").strip()
            messages[0] = (init_role, f"{prompt_token}\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""
            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.LLAMA_3:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_3_1:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.QWEN2:  # fix: add qwen2
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.QWEN2VL:  # fix: add qwen2
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.GEMMA:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += "<start_of_turn>" + role + "\n" + message + "<end_of_turn>\n" + seps[i % 2]
                else:
                    ret += "<start_of_turn>" + role + "\n"
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        elif self.sep_style == SeparatorStyle.SPEECH_PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(self, image, image_process_mode, return_pil=False, image_format='PNG', max_len=1344, min_len=672):
        if image_process_mode == "Pad":
            def expand2square(pil_img, background_color=(122, 116, 104)):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
        if max(image.size) > max_len:
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
            longest_edge = int(shortest_edge * aspect_ratio)
            W, H = image.size
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, speech, image, video, image_process_mode = msg
                    images.append(image)
        return images
    
    def get_speeches(self):
        speeches = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, speech, image, video, image_process_mode = msg
                    speeches.append(speech)
        print(speeches)
        return speeches
    
    def get_videos(self):
        videos = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, speech, image, video, image_process_mode = msg
                    videos.append(video)
        print(videos)
        return videos

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, speech, image, video, image_process_mode = msg
                    if video is not None:
                        base64_string = mov_to_base64(video)
                        video_str = f'''
                        <video width="300" controls>
                            <source src="data:video/quicktime;base64,{base64_string}" type="video/quicktime">
                            Your browser does not support the video element.
                        </video>
                        '''
                        msg = video_str + msg.replace('<image>', '').strip().replace('<speech>', '').strip()
                        ret.append([msg, None])
                    elif image is not None:
                        image = Image.open(image)
                        img_b64_str = self.process_image(
                            image, "Default", return_pil=False,
                            image_format='JPEG')
                        img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                        if speech is not None:
                            base64_string = wav_to_base64(speech)
                            aud_str = f'''
                            <audio controls>
                                <source src="data:audio/wav;base64,{base64_string}" type="audio/wav">
                                Your browser does not support the audio element.
                            </audio>
                            '''
                            msg = img_str + aud_str + msg.replace('<image>', '').strip().replace('<speech>', '').strip()
                            ret.append([msg, None])
                        else:
                            msg = img_str + msg.replace('<image>', '').strip()
                            ret.append([msg, None])
                    elif speech is not None:
                        base64_string = wav_to_base64(speech)
                        aud_str = f'''
                        <audio controls>
                            <source src="data:audio/wav;base64,{base64_string}" type="audio/wav">
                            Your browser does not support the audio element.
                        </audio>
                        '''
                        msg = aud_str + msg.replace('<speech>', '').strip()
                        ret.append([msg, None])
                    else:
                        ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                if type(msg) is tuple and len(msg) == 2:
                    msg, img_b64_str = msg
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                    msg = msg.strip() + img_str
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llama_3 = Conversation(
    system="<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("<|start_header_id|>user<|end_header_id|>\n\n", 
           "<|start_header_id|>system<|end_header_id|>\n\n"),
    version="llama_v3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_3,
    sep="<|eot_id|>",
)


conv_llama_3_1 = Conversation(
    system="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("<|start_header_id|>user<|end_header_id|>\n\n", 
           "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    version="llama_v31",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_3_1,
    sep="<|eot_id|>",
)


conv_qwen = Conversation(
    system="""<|im_start|>system\nYou are a helpful assistant.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="qwen",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.QWEN2,
    sep="<|im_end|>\n",
)

conv_qwen2vl = Conversation(
    system="""<|im_start|>system\nYou are a helpful assistant.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="qwen",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.QWEN2VL,
    sep="<|im_end|>\n",
)


conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

speech_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SPEECH_PLAIN,
    sep="\n",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_vicuna_imgsp_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="imgsp_v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

conv_phi_2 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="phi2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|endoftext|>",
)

conv_mistral_instruct = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_gemma = Conversation(
    system="",
    roles=("user", "model"),
    version="gemma",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.GEMMA,
    sep="",
    sep2="<eos>",
)

conv_chatml_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

default_conversation = conv_qwen2vl
conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "phi_2": conv_phi_2,
    "gemma": conv_gemma,
    "llama_2": conv_llama_2,
    "llama_3": conv_llama_3,
    "llama_3_1": conv_llama_3_1,
    "qwen2": conv_qwen,
    "qwen2vl": conv_qwen2vl,
    "imgsp_v1": conv_vicuna_imgsp_v1,
    "mistral_instruct": conv_mistral_instruct,
    "chatml_direct": conv_chatml_direct,
    "mistral_direct": conv_chatml_direct,
    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "speech_plain": speech_plain,
    "llava_v0": conv_llava_v0,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,
    "mpt": conv_mpt,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())