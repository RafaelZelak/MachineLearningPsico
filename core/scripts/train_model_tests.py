import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

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
    
    # O output precisa ser um array de inteiros (os índices das palavras)
    output_sequences = np.array(output_sequences)
    
    # Criar o modelo de resposta com camadas adicionais e Dropout
    model = Sequential()
    model.add(Embedding(total_words, 200, input_length=max_input_len))  # Embedding de 200 dimensões
    model.add(LSTM(200, return_sequences=True))  # Aumentando as células LSTM
    model.add(Dropout(0.2))  # Dropout para evitar overfitting
    model.add(LSTM(200))  # Segunda camada LSTM
    model.add(Dense(total_words, activation='softmax'))  # Camada final com softmax
    
    # Compilar o modelo com um otimizador Adam e uma taxa de aprendizado menor
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Treinar o modelo
    X_train, X_test, y_train, y_test = train_test_split(input_sequences, output_sequences, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1, validation_data=(X_test, y_test))
    
    return model, tokenizer, max_input_len

# Função para classificar e gerar uma resposta com base na frase e no diagnóstico
def generate_response(diagnostico, frase, model, tokenizer, max_input_len):
    # Combinar o diagnóstico com a frase como entrada
    input_text = f"{diagnostico} {frase}"
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_input_len, padding='pre')
    
    # Gerar uma resposta palavra por palavra
    predicted_sequence = []
    
    for _ in range(20):  # Limite de 20 palavras para a resposta gerada
        predicted_probs = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted_probs, axis=-1)

        # Converter o índice previsto para palavra
        predicted_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted[0]:
                predicted_word = word
                break
        
        if predicted_word == '':
            break  # Parar se nenhuma palavra for prevista
        
        predicted_sequence.append(predicted_word)
        
        # Atualizar a entrada com a palavra prevista
        predicted = np.expand_dims(predicted, axis=-1)  # Ajusta predicted para ter duas dimensões
        token_list = np.append(token_list[:, 1:], predicted, axis=1)
    
    # Juntar a sequência prevista em uma string
    return ' '.join(predicted_sequence)

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
