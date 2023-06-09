import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

st.title('AI Image Prompt Generator')

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained('FredZhang7/distilgpt2-stable-diffusion-v2')

prompt = st.text_input('Enter a subject or item')

if st.button('Generate'):
    temperature = 0.9
    top_k = 8
    max_length = 80
    repetition_penalty = 1.2
    num_return_sequences = 5

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(
        input_ids,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        repetition_penalty=repetition_penalty,
        early_stopping=True
    )

    generated_texts = []
    for i in range(len(output)):
        generated_texts.append(tokenizer.decode(output[i], skip_special_tokens=True))

    st.subheader('Generated Prompts')
    for text in generated_texts:
        st.write(text)
        st.write('---')

