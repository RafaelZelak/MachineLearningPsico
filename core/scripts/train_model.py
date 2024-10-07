import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Função para treinar o modelo e classificá-lo
def train_model():
    data = pd.read_csv('core/info_psico.csv')
    X = data['frase']  # Frases de entrada
    y = data['diagnostico']  # Diagnósticos (depressão, ansiedade, etc.)

    # Criar pipeline para vetorizar o texto e treinar o modelo Naive Bayes
    model = make_pipeline(CountVectorizer(), MultinomialNB())

    # Dividir os dados em conjunto de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinar o modelo com os dados de treino
    model.fit(X_train, y_train)

    return model

# Função para classificar a mensagem e retornar o ID do transtorno
def classify_message_with_ai(mensagem):
    # Treinar o modelo (ou carregar o modelo treinado, se já existir)
    model = train_model()

    # Fazer a previsão para a mensagem recebida
    predicted_label = model.predict([mensagem])[0]  # Retorna 'depressão', 'ansiedade', etc.

    # Mapeamento de transtornos para IDs
    diagnostico_map = {
        'depressão': 1,
        'ansiedade': 2,
        'pânico': 3,
        'burnout': 4
    }

    # Retorna o ID correspondente ao transtorno previsto
    return diagnostico_map.get(predicted_label.lower(), None)  # 'None' se o transtorno não for encontrado
