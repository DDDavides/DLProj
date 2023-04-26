import torch
from dalle_pytorch import OpenAIDiscreteVAE, DALLE
from tqdm import tqdm
import argparse
from pathlib import Path
from tqdm import tqdm

# torch

import torch

from einops import repeat

# vision imports

from PIL import Image
from torchvision.utils import make_grid, save_image

# dalle related classes and utils

from dalle_pytorch import __version__
from dalle_pytorch import DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE, DALLE
from dalle_pytorch.tokenizer import SimpleTokenizer, HugTokenizer, YttmTokenizer, ChineseTokenizer

vae = OpenAIDiscreteVAE()       # loads pretrained OpenAI VAE

dalle = DALLE(
    dim = 1024,
    vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
    num_text_tokens = 10000,    # vocab size for text
    text_seq_len = 256,         # text sequence length
    depth = 1,                  # should aim to be 64
    heads = 16,                 # attention heads
    dim_head = 64,              # attention head dimension
    attn_dropout = 0.1,         # attention dropout
    ff_dropout = 0.1            # feedforward dropout
)

import argparse
from pathlib import Path
from tqdm import tqdm

# torch

import torch

from einops import repeat

# vision imports

from PIL import Image
from torchvision.utils import make_grid, save_image

# dalle related classes and utils

from dalle_pytorch import __version__
from dalle_pytorch import DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, YttmTokenizer, ChineseTokenizer

# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--dalle_path', type = str, required = False,
                    help='path to your trained DALL-E')

parser.add_argument('--vqgan_model_path', type=str, default = None,
                   help='path to your trained VQGAN weights. This should be a .ckpt file. (only valid when taming option is enabled)')

parser.add_argument('--vqgan_config_path', type=str, default = None,
                   help='path to your trained VQGAN config. This should be a .yaml file.  (only valid when taming option is enabled)')

parser.add_argument('--text', type = str, required = True,
                    help='your text prompt')

parser.add_argument('--num_images', type = int, default = 128, required = False,
                    help='number of images')

parser.add_argument('--batch_size', type = int, default = 4, required = False,
                    help='batch size')

parser.add_argument('--top_k', type = float, default = 0.9, required = False,
                    help='top k filter threshold')

parser.add_argument('--outputs_dir', type = str, default = './outputs', required = False,
                    help='output directory')

parser.add_argument('--bpe_path', type = str,
                    help='path to your huggingface BPE json file')

parser.add_argument('--hug', dest='hug', action = 'store_true')

parser.add_argument('--chinese', dest='chinese', action = 'store_true')

parser.add_argument('--taming', dest='taming', action='store_true')

parser.add_argument('--gentxt', dest='gentxt', action='store_true')

args = parser.parse_args()

# helper fns

def exists(val):
    return val is not None

# tokenizer

if exists(args.bpe_path):
    klass = HugTokenizer if args.hug else YttmTokenizer
    tokenizer = klass(args.bpe_path)
elif args.chinese:
    tokenizer = ChineseTokenizer()

# generate images

image_size = vae.image_size

texts = args.text.split('|')

for j, text in tqdm(enumerate(texts)):
    if args.gentxt:
        text_tokens, gen_texts = dalle.generate_texts(tokenizer, text=text, filter_thres = args.top_k)
        text = gen_texts[0]
    else:
        text_tokens = tokenizer.tokenize([text], dalle.text_seq_len)

    text_tokens = repeat(text_tokens, '() n -> b n', b = args.num_images)

    outputs = []

    for text_chunk in tqdm(text_tokens.split(args.batch_size), desc = f'generating images for - {text}'):
        output = dalle.generate_images(text_chunk, filter_thres = args.top_k)
        outputs.append(output)

    outputs = torch.cat(outputs)

    # save all images

    file_name = text 
    outputs_dir = Path(args.outputs_dir) / file_name.replace(' ', '_')[:(100)]
    outputs_dir.mkdir(parents = True, exist_ok = True)

    for i, image in tqdm(enumerate(outputs), desc = 'saving images'):
        save_image(image, outputs_dir / f'{i}.png', normalize=True)
        with open(outputs_dir / 'caption.txt', 'w') as f:
            f.write(file_name)

    print(f'created {args.num_images} images at "{str(outputs_dir)}"')