from transformers import set_seed
from kgen_exp.diff import load_model, generate, encode_prompts


PROMPTS = """
1girl,
grass wonder (umamusume), umamusume,
ninjin nouka, ask (askzy), kita (kitairoha), 

solo, wide hips, bathroom, swimsuit, breasts, open mouth, horse ears, tail, thighs, 
brown hair, shower \(place\), towel on head, horse tail, white bikini, cleavage, 
navel, side-tie bikini bottom, showering, horse girl, blue eyes, wet, smile, 
collarbone, blush, towel, armpits, large breasts, soap, bikini, indoor swimsuit, 
animal ears, looking at viewer, bare shoulders, indoors, long hair, arm up,

An illustration of a young woman standing in front of a shower. 
she has blue eyes and is looking off to the side with a peaceful expression on her face. 
she is wearing a white towel wrapped around her head and has long brown hair that is styled in loose waves. 
the woman is holding a yellow sponge in her right hand and appears to be brushing her teeth. 
the showerhead is visible in the background, and there are raindrops falling around her. 
the overall mood of the image is calm and serene.

masterpiece, newest, absurdres, sensitive
""".strip()
NEGATIVE = """
worst quality
""".strip()


pipe = load_model("KBlueLeaf/Kohaku-XL-Zeta", "cuda")


if __name__ == "__main__":
    for pad in [True, False]:
        for eos in [True, False]:
            set_seed(0)
            (prompt_embeds, neg_prompt_embeds), (pooled_embeds2, neg_pooled_embeds2) = (
                encode_prompts(
                    pipe,
                    PROMPTS,
                    NEGATIVE,
                    cutoff_length=300,
                    padding_to_max_length=pad,
                    take_all_eos=eos,
                )
            )
            print(pad, eos)
            print(prompt_embeds.shape, neg_prompt_embeds.shape)
            print(pooled_embeds2.shape, neg_pooled_embeds2.shape)
            images = generate(
                pipe,
                prompt_embeds,
                neg_prompt_embeds,
                pooled_embeds2,
                neg_pooled_embeds2,
                24,
                832,
                1216,
                5,
            )
            images[0].save(f"./output/kxl-test-{pad=}-{eos=}.png")
