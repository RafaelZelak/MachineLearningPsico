import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
import nltk
from tensorflow.keras.utils import to_categorical  # Adiciona a função de one-hot encoding

nltk.download('punkt')
from nltk.tokenize import word_tokenize
print("GPUs disponíveis: ", tf.config.list_physical_devices('GPU'))

# Função para processar os dados e treinar o modelo de geração de resposta
def train_response_generation_model():
    # Carregar o dataset
    data = pd.read_csv('core/info_psico.csv')
    
    # Concatenar o diagnóstico e a frase para formar a entrada do modelo
    data['input'] = data['diagnostico'] + ' ' + data['frase']
    
    # Tokenizar o texto das entradas e das respostas
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['input'].values.tolist() + data['resposta'].values.tolist())
    
    total_words = len(tokenizer.word_index) + 1
    
    # Preparar sequências de entrada e saída
    input_sequences = []
    output_sequences = []
    
    for i, row in data.iterrows():
        input_sequence = tokenizer.texts_to_sequences([row['input']])[0]
        output_sequence = tokenizer.texts_to_sequences([row['resposta']])[0]
        
        # Criar múltiplas sequências de treino para gerar uma palavra por vez
        for j in range(1, len(output_sequence)):
            input_sequences.append(input_sequence)  # Entrada completa
            output_sequences.append(output_sequence[j])  # Próxima palavra na sequência
    
    # Padronizar as sequências de entrada para que tenham o mesmo comprimento
    max_input_len = max([len(x) for x in input_sequences])
    
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_input_len, padding='pre'))
    
    # O output precisa ser convertido para one-hot encoding
    output_sequences = np.array(output_sequences)
    output_sequences = to_categorical(output_sequences, num_classes=total_words)  # One-hot encoding
    
    # Criar o modelo de resposta com camadas adicionais e Dropout
    model = Sequential()
    model.add(Embedding(total_words, 600, input_length=max_input_len))
    model.add(Bidirectional(LSTM(300, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(300)))
    model.add(Dense(total_words, activation='softmax'))
    
    # Compilar o modelo com CategoricalCrossentropy
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer='adam',
        metrics=['accuracy']
    )
    
    # Treinar o modelo
    X_train, X_test, y_train, y_test = train_test_split(input_sequences, output_sequences, test_size=0.2, random_state=42)
    
    with tf.device('/GPU:0'):
        model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test))
    
    return model, tokenizer, max_input_len

# Função para classificar e gerar uma resposta com base na frase e no diagnóstico
def generate_response(diagnostico, frase, model, tokenizer, max_input_len, beam_width=3):
    input_text = f"{diagnostico} {frase}"
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_input_len, padding='pre')
    
    beam = [(token_list, [], 0)]  # (sequence, output words, cumulative log prob)
    
    for _ in range(20):  # Limite de 20 palavras
        all_candidates = []
        for seq, out_words, log_prob in beam:
            predicted_probs = model.predict(seq, verbose=0)
            top_candidates = np.argsort(predicted_probs[0])[-beam_width:]
            for word_idx in top_candidates:
                word = tokenizer.index_word.get(word_idx, '')
                if word:
                    all_candidates.append((np.append(seq[:, 1:], [[word_idx]], axis=1), out_words + [word], log_prob + np.log(predicted_probs[0][word_idx])))
        
        beam = sorted(all_candidates, key=lambda x: x[2], reverse=True)[:beam_width]
        
    return ' '.join(beam[0][1])

# Execução isolada
if __name__ == "__main__":
    # Treinar o modelo de geração de respostas
    model, tokenizer, max_input_len = train_response_generation_model()
    
    # Teste de classificação e geração de resposta
    diagnostico_teste = 'ansiedade'
    frase_teste = "Estou me sentindo muito ansioso ultimamente."
    
    resposta_gerada = generate_response(diagnostico_teste, frase_teste, model, tokenizer, max_input_len)
    
    # Exibir o resultado
    print(f"Resposta sugerida: {resposta_gerada}")
