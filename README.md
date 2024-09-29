# KGen - A System for Prompt Generation to Improve Text-to-Image Performance

KGen is a project that utilizes Large Language Models (LLMs) to generate prompts for Text-to-Image (T2I) models.

The goal is to enable T2I models to use more complicated and detailed captions during training while maintaining good usability.

## Usage

Installation:

```bash
pip install tipo-kgen
```

Use in code:
Read the [Example code](scripts/example.py) or [TIPO-test script](scripts/tipo-test.py) for more informations.

## TIPO

TIPO: Text to Image with text Presampling for Optimal prompting

TIPO is a LLM model system designed for generating detailed prompt from input tags or caption. Unlike DTG, TIPO can handle both tags and Natural language. In theory, you can also design your own tag in linguistic way. (For example, long blue hair is acceptable tag in TIPO and will not break the model).
The main difference between TIPO and DTG is:

1. TIPO is trained with both Natural Language captions and Danbooru tags, the "nl+tags" data are also not only from danbooru but also some general text-image dataset like Coyo-11M
2. TIPO is trained with better format which achieve some ability on "generate meta infos" such as artists/characters. (or, you can say TIPO have ability to "choose" which artist tag is suitable with current content)
3. TIPO is trained on 30M entries dataset, with more than 25M entries have NL caption, more than 18M entries have tags

### Model card

|                   | TIPO-200M                                                                      | TIPO-500M                                                                      |
| ----------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Arch              | LLaMA                                                                          | LLaMA                                                                          |
| Max ctx length    | 1024                                                                           | 1024                                                                           |
| Batch Size        | 2048                                                                           | 3584                                                                           |
| Training dataset  | Danbooru, GBC10M, 5epoch<br />Danbooru, GBC10M, Coyo11M, 3epoch              | Danbooru, GBC10M, Coyo11M, 5epoch                                            |
| Real Token Seen*  | 40B token                                                                      | 30B token                                                                      |
| Training Hardware | RTX 3090 x 4                                                                   | H100 x 8                                                                       |
| Training Time     | 420 hour`                                                                      | 100 hour`                                                                      |
| Huggingface       | [KBlueLeaf/TIPO-200M · Hugging Face](https://huggingface.co/KBlueLeaf/TIPO-200M) | [KBlueLeaf/TIPO-500M · Hugging Face](https://huggingface.co/KBlueLeaf/TIPO-500M) |

*: We only count "non-padding token" in the token seen, since all the training data have very large length range.`<br/>`
`: Since the training data is pretty short, it cost more time to reach same token seen than general LLM pretraining.`<br/>`
As reference, with 4096 as max ctx length and almost all the data have reach that length, you may only need 2days to reach 10B token seen on RTX 3090 x 4 with 200M model.

### Usage

A Simple DEMO for TIPO (with t2i functionality included):
https://huggingface.co/spaces/KBlueLeaf/TIPO-DEMO

TIPO-extension: https://github.com/KohakuBlueleaf/z-tipo-extension

## DanTagGen

DanTagGen is an early project under KGen, trained on the Danbooru tag system. Danbooru tags often have "overlaps" or "duplicates", such as:

- "long hair" and "very long hair"
- "thighhighs", "black thighhighs", and "black legwears"

Although users naturally avoid these duplications, the model may benefit from having the complete set of tags for better results, as that is how they were trained.

In addition to overlapping tags, "character tags" also need to be mentioned. For simplicity, a "DreamBooth style" prompt can be used to illustrate this:

- Original: a dog
- DreamBooth: a \[V\] dog
- What User wants: \[V\]

As shown above, users tend to ignore all the "descriptions" that directly point to the character. In this situation, utilizing LLMs to "connect" the "trigger word" and "related description" is a promising approach. DanTagGen is an experimental project to prove this concept.

### Architecture

DanTagGen uses the LLaMA architecture with 400M parameters.

### Training

DanTagGen is trained on posts with the top 75% favorite count in Danbooru, which amounts to 5 million entries.

More details about the architecture and training can be found on the Hugging Face page: [KBlueLeaf/DanTagGen-beta · Hugging Face](https://huggingface.co/KBlueLeaf/DanTagGen-beta)

### Usage

* Hugging Face Space: [DTG Demo - a Hugging Face Space by KBlueLeaf](https://huggingface.co/spaces/KBlueLeaf/DTG-demo)
* SD-WebUI Extension: [KohakuBlueleaf/z-a1111-sd-webui-dtg: A sd-webui extension for utilizing DanTagGen to &#34;upsample prompts&#34;. (github.com)](https://github.com/KohakuBlueleaf/z-a1111-sd-webui-dtg)
* ComfyUI Node: [toyxyz/a1111-sd-webui-dtg_comfyui: A sd-webui extension for utilizing DanTagGen to &#34;upsample prompts&#34;. (github.com)](https://github.com/toyxyz/a1111-sd-webui-dtg_comfyui)
