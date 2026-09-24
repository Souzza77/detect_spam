import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

st.set_page_config(page_title="DM Shield - Detector de Spam", page_icon="🛡️", layout="centered")

# ---------------- Dataset de treinamento (exemplos de DMs) ----------------
SPAM_MESSAGES = [
    "Parabéns! Você ganhou um iPhone, clique aqui para resgatar: bit.ly/premio123",
    "Invista em criptomoedas e multiplique seu dinheiro em 24h, clique no link",
    "Sua conta será suspensa, verifique seus dados agora: link-falso.com/verificar",
    "Oi lindx, vi seu perfil e quero te enviar um presente, me chama no whatsapp",
    "Ganhe seguidores grátis agora mesmo, acesse o link na bio",
    "Você foi selecionado para uma parceria exclusiva, envie seus dados bancários",
    "Clique aqui para confirmar seu prêmio antes que expire: bit.ly/xyz",
    "Empresa de apostas paga R$500 só para se cadastrar, link no perfil",
    "Sua senha foi comprometida, faça login aqui para proteger sua conta",
    "Trabalhe de casa e ganhe R$3000 por semana, mande seu whatsapp",
    "Duplique seu investimento em 48 horas, fale comigo agora",
    "Recebi seu contato de um amigo, tenho uma proposta imperdível, clique aqui",
    "Verifique sua conta do Instagram agora ou ela será deletada em 24h",
    "Parabéns, você foi sorteado! Preencha o formulário com seus dados",
    "Ganhe dinheiro fácil compartilhando este link com seus seguidores",
    "Oferta exclusiva só hoje, deposite e receba o dobro amanhã",
    "Confirma seu CPF aqui para não perder o selo verificado",
    "Amor, preciso de ajuda urgente, me envia um pix agora",
]

SAFE_MESSAGES = [
    "Oi, adorei seu último post, você é uma inspiração",
    "Podemos marcar uma reunião para discutir uma parceria de marca?",
    "Meu nome é Ana, trabalho na agência X e gostaria de propor uma collab",
    "Seu vídeo me ajudou muito, obrigado por compartilhar",
    "Qual câmera você usa para gravar seus vídeos?",
    "Adorei o conteúdo de hoje, muito bem explicado",
    "Estou organizando um evento e gostaria de te convidar",
    "Sou fã do seu trabalho há anos, continue assim",
    "Poderia me indicar onde comprar aquele produto que você usou?",
    "Parabéns pelo aniversário do canal, sucesso sempre",
    "Gostaria de saber mais sobre seu processo criativo",
    "Sua live de ontem foi ótima, aprendi bastante",
    "Somos uma marca de roupas e queremos enviar produtos para você testar",
    "Oi, sou jornalista e gostaria de fazer uma entrevista com você",
    "Muito bom o conteúdo, você recomenda algum curso sobre o tema?",
    "Trabalho com edição de vídeo, adoraria colaborar com você",
    "Vi que você mora perto de mim, adorei o passeio que mostrou",
    "Seu último carrossel ficou muito bem produzido, parabéns pela equipe",
]

@st.cache_resource(show_spinner="Treinando modelo de classificação...")
def load_model():
    texts = SPAM_MESSAGES + SAFE_MESSAGES
    labels = np.array([1] * len(SPAM_MESSAGES) + [0] * len(SAFE_MESSAGES), dtype="float32")

    vectorizer = layers.TextVectorization(max_tokens=2000, output_sequence_length=30)
    vectorizer.adapt(texts)

    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    x = vectorizer(inputs)
    x = layers.Embedding(input_dim=2000, output_dim=16, mask_zero=True)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(tf.constant(texts, dtype=tf.string), labels, epochs=150, batch_size=8, verbose=0)
    return model

model = load_model()

# ---------------- Interface ----------------
st.title("🛡️ DM Shield")
st.caption("Ferramenta interna para triagem de mensagens diretas suspeitas")

st.markdown("---")

message = st.text_area(
    "Cole a mensagem recebida:",
    height=150,
    placeholder="Ex: Parabéns! Você ganhou um prêmio, clique aqui...",
)

col1, col2 = st.columns([1, 3])
with col1:
    check = st.button("Verificar mensagem", type="primary", use_container_width=True)

if check:
    if not message.strip():
        st.warning("Cole uma mensagem antes de verificar.")
    else:
        pred = float(model.predict(tf.constant([message], dtype=tf.string), verbose=0)[0][0])
        is_spam = pred >= 0.5
        confidence = pred if is_spam else 1 - pred

        st.markdown("### Resultado")
        if is_spam:
            st.error(f"🚨 SPAM — confiança de {confidence:.0%}")
            st.write(
                "Esta mensagem apresenta características comuns de golpes: promessas "
                "de prêmios, urgência, links suspeitos ou pedido de dados pessoais."
            )
        else:
            st.success(f"✅ SEGURA — confiança de {confidence:.0%}")
            st.write("Não foram identificados padrões típicos de spam ou golpe nesta mensagem.")

        with st.expander("Detalhes técnicos"):
            st.write(f"Score do modelo: {pred:.4f}")
            st.write(
                "Modelo: TensorFlow (Embedding + Dense) treinado com exemplos de "
                "DMs reais de golpes e mensagens legítimas."
            )

st.markdown("---")
st.caption(
    "Uso interno — Sistema de triagem automática de mensagens. "
    "Este resultado é uma sugestão e não substitui a análise humana."
)