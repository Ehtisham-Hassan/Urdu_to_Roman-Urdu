import streamlit as st
import torch
import pickle
import os
import gdown
from model import Seq2SeqModel

# ==============================
# Google Drive File IDs
# ==============================
# Replace these with actual file IDs from your Drive folder
MODEL_FILE_ID = "1aFHD_0Vmr8imoL6Rg_RIdtCee0yQdopG"  # best_model.pth
SRC_TOKENIZER_FILE_ID = "1AO1Tzen7N82ScjFoCVrWb-717eGqXUB3"  # src_tokenizer.pkl
TGT_TOKENIZER_FILE_ID = "1gt3VRG58U8dct3-QROA9OFeKIF5Pz6Kr"  # tgt_tokenizer.pkl

files_to_download = {
    "best_model.pth": MODEL_FILE_ID,
    "src_tokenizer.pkl": SRC_TOKENIZER_FILE_ID,
    "tgt_tokenizer.pkl": TGT_TOKENIZER_FILE_ID,
}


# ==============================
# Download files if missing
# ==============================
for filename, file_id in files_to_download.items():
    if not os.path.exists(filename):
        url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        st.write(f"📥 Downloading {filename} from Google Drive...")
        gdown.download(url, filename, quiet=False)

# ==============================
# Load tokenizers
# ==============================
with open("src_tokenizer.pkl", "rb") as f:
    src_tokenizer = pickle.load(f)
with open("tgt_tokenizer.pkl", "rb") as f:
    tgt_tokenizer = pickle.load(f)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# Load model
# ==============================
model = Seq2SeqModel(
    src_vocab_size=len(src_tokenizer.vocab),
    tgt_vocab_size=len(tgt_tokenizer.vocab),
    embed_dim=256,
    hidden_dim=512,
    encoder_layers=2,
    decoder_layers=4,
    dropout=0.3
).to(device)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ==============================
# Translation function
# ==============================
def translate_sentence(sentence, max_len=50):
    src_tokens = src_tokenizer.encode(sentence)
    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
    src_lengths = torch.tensor([len(src_tokens)], dtype=torch.long).to(device)

    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(src_tensor, src_lengths)

        mask = torch.zeros(1, src_tensor.size(1), device=device)
        mask[0, :len(src_tokens)] = 1

        hidden = model.decoder.hidden_projection(hidden).unsqueeze(0).repeat(model.decoder.num_layers, 1, 1)
        cell = model.decoder.cell_projection(cell).unsqueeze(0).repeat(model.decoder.num_layers, 1, 1)

        input_token = torch.tensor([[tgt_tokenizer.vocab["<sos>"]]], device=device)

        output_tokens = []
        for _ in range(max_len):
            output, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs, mask)
            top1 = output.argmax(2)
            if top1.item() == tgt_tokenizer.vocab["<eos>"]:
                break
            output_tokens.append(top1.item())
            input_token = top1

    return tgt_tokenizer.decode(output_tokens)

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Urdu → Roman Urdu Translator", page_icon="🌐")
st.title("🌐 Urdu → Roman Urdu Translator")
st.write("Neural Machine Translation using BiLSTM + Attention")

user_input = st.text_area("Enter Urdu text here:", "")

if st.button("Translate"):
    if user_input.strip():
        translation = translate_sentence(user_input)
        st.success(translation)
    else:
        st.warning("⚠️ Please enter some Urdu text to translate.")
