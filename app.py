from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from core.scripts.train_model import classify_message_with_ai

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Função para conectar ao banco de dados SQLite com timeout para evitar "database is locked"
def get_db_connection():
    conn = sqlite3.connect('data/user_data.db', timeout=10)  # Timeout de 10 segundos
    conn.row_factory = sqlite3.Row
    return conn

# Página inicial - tela de login
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email_or_name = request.form['email_or_name']
        password = request.form['password']
        conn = get_db_connection()
        try:
            user = conn.execute("SELECT * FROM user WHERE email = ? OR nome = ?", (email_or_name, email_or_name)).fetchone()
        finally:
            conn.close()

        if user and check_password_hash(user['senha'], password):
            session['user_id'] = user['id']
            session['nome'] = user['nome']
            return redirect(url_for('dashboard'))
        else:
            flash('Login inválido. Verifique suas credenciais.')
            return redirect(url_for('login'))
    return render_template('login.html')

# Página de criação de conta
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        nome = request.form['nome']
        email = request.form['email']
        idade = request.form['idade']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('As senhas não coincidem!')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO user (nome, email, idade, senha, data_criacao) VALUES (?, ?, ?, ?, ?)",
                         (nome, email, idade, hashed_password, datetime.now()))
            conn.commit()
        finally:
            conn.close()

        flash('Conta criada com sucesso! Faça login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        flash('Faça login para acessar o dashboard.')
        return redirect(url_for('login'))

    diagnostico_map = {
        1: 'Depressão',
        2: 'Ansiedade',
        3: 'Pânico',
        4: 'Burnout'
    }

    if request.method == 'POST':
        mensagem = request.form['mensagem'].strip()

        # Verificar se a mensagem está vazia
        if not mensagem:
            flash('A mensagem não pode estar vazia.')
            return redirect(url_for('dashboard'))

        # Classificar a mensagem com o modelo de IA
        id_diagnostico = classify_message_with_ai(mensagem)

        if id_diagnostico is None:
            flash('Não foi possível identificar um diagnóstico.')
            return redirect(url_for('dashboard'))

        # Salvar os dados no banco de dados
        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO chat_memory (id_user, mensagem, id_diagnostico, data_criacao) VALUES (?, ?, ?, ?)",
                         (session['user_id'], mensagem, id_diagnostico, datetime.now()))
            conn.commit()
        finally:
            conn.close()

        flash('Dados salvos com sucesso!')
        return redirect(url_for('dashboard'))

    # Recuperar as últimas mensagens e diagnósticos do usuário para exibir no dashboard
    conn = get_db_connection()
    chat_memory = conn.execute("SELECT * FROM chat_memory WHERE id_user = ?", (session['user_id'],)).fetchall()
    conn.close()

    return render_template('dashboard.html', nome=session['nome'], chat_memory=chat_memory, diagnostico_map=diagnostico_map)



# Logout
@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    flash('Logout realizado com sucesso.')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
