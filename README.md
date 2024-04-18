# KGen - A System for Prompt Generation to Improve Text-to-Image Performance

***\*WIP\****

KGen is a project that utilizes Large Language Models (LLMs) to generate prompts for Text-to-Image (T2I) models. 

The goal is to enable T2I models to use more complicated and detailed captions during training while maintaining good usability.

## Usage

Installation:

```bash
pip install tipo-kgen
```

Use in code:
Read the [Example code](example.py) for more informations.

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

* Hugging Face Space:
* SD-WebUI Extension: [KohakuBlueleaf/z-a1111-sd-webui-dtg: A sd-webui extension for utilizing DanTagGen to &#34;upsample prompts&#34;. (github.com)](https://github.com/KohakuBlueleaf/z-a1111-sd-webui-dtg)
* ComfyUI Node:
