# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 01:54:46 2024

@author: adria
"""

#importaciones libreria

import os
import random
import string
import joblib
import re
import nltk
import pandas as pd
import numpy as np
import spacy

#llamadas de modulos librerias

from nltk.stem import SnowballStemmer
from nltk.chat.util import Chat, reflections
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.chat.util import Chat, reflections
from nltk.tokenize import word_tokenize
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline

#variables globales
tokenizer = None
X = None

data_file='datos2.csv'
entrenadores="entrenadores.csv"
model_file = 'rnn_model.h5'
weights_file = 'rnn_model_weights.h5'
tokenizer_file = 'tokenizer.pkl'
label_encoder_file = 'label_encoder.pkl'
configuracion_file = 'configuracion.csv'
MAX_SEQUENCE_LENGTH = 4  # Puedes ajustar este valor según tus necesidades

#descargas de componentes
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Definir lematizador y stemmer para español
lemmatizer_es = WordNetLemmatizer()
stop_words_es = set(stopwords.words('spanish'))
stemmer_es = SnowballStemmer('spanish')

# Definir lematizador y stemmer para inglés
lemmatizer_en = WordNetLemmatizer()
stop_words_en = set(stopwords.words('english'))
stemmer_en = SnowballStemmer('english')

#cargar archivo de configuracion
def load_configuracion():
    try:
        return pd.read_csv(configuracion_file)
    except FileNotFoundError:
        return pd.DataFrame(columns=['codigo', 'tipo', 'nombre'])
#carga de datos
def load_data(data_file):
    return pd.read_csv(data_file)
#guardar datos
def save_data(data, data_file):
    data.to_csv(data_file, index=False)
    

 #preprocesado ingles
def preprocess_ingles(text, language='english'):
     text = text.lower()
     text = text.translate(str.maketrans('', '', string.punctuation.replace('\'', '')))
     
     stop_words = set(stopwords.words(language))
     stemmer = SnowballStemmer(language)
     
     tokens = nltk.word_tokenize(text)
     tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
     return ' '.join(tokens)

#preprocesado español
def preprocess_spanish(text, language='spanish'):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation.replace('\'', '')))

    stop_words = set(stopwords.words(language))
    lemmatizer = WordNetLemmatizer()  # Utiliza un lematizador en lugar de un stemmer

    tokens = word_tokenize(text, language='spanish')
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

#categoria gramatical ingles
def get_wordnet_pos_ingles(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

#categoria gramatical español
def get_wordnet_pos_español(word):
    # Cargar el modelo en español de spaCy
    nlp = spacy.load("es_core_news_sm")

    # Etiquetar la parte del discurso (POS) de la palabra
    try:
        pos_tag = nlp(word)[0].pos_
    except IndexError:
        # Manejar la excepción si no se encuentra información POS
        pos_tag = 'NOUN'

    # Mapear las etiquetas POS a las categorías de WordNet en español (puedes ajustar según sea necesario)
    tag_dict_es = {"ADJ": "ADJ", "NOUN": "NOUN", "VERB": "VERB", "ADV": "ADV"}

    # Devolver la categoría gramatical correspondiente o 'NOUN' por defecto
    return tag_dict_es.get(pos_tag, 'NOUN')


#agregar la nueva entrada

def add_new_entry(data, entry, response, nombre_usuario, configuracion):
    train_model(data)

    # Obtener la fecha y hora actual
    fecha_hora_actual = datetime.now().strftime("%Y%m%d,%H:%M:%S")

    # Crear la nueva entrada
    new_entry = pd.DataFrame({
        'fecha': [fecha_hora_actual],
        'hora': [fecha_hora_actual.split(',')[1]],
        'tipo': ['informacion'],
        'entrada': [entry],
        'salida': [response],
        'tipo_usuario': [configuracion['tipo'].iloc[0] if not configuracion.empty else None],
        'nombre_usuario': [nombre_usuario]
    })

    # Concatenar la nueva entrada al conjunto de datos
    data = pd.concat([data, new_entry], ignore_index=True)

    return data

#actualizar entrada
def update_entry(data, entry, response, configuracion):
    # Obtener la fila correspondiente a la entrada actual
    entry_row = data[data['entrada'] == entry]

    if not entry_row.empty:
        # Obtener información del usuario actual desde configuracion
        tipo_usuario = configuracion['tipo'].iloc[0] if not configuracion.empty else None
        nombre_usuario = configuracion['nombre'].iloc[0] if 'nombre' in configuracion.columns else None

        # Obtener la fecha y hora actual
        fecha_hora_actual = datetime.now().strftime("%Y%m%d,%H:%M:%S")

        # Actualizar los campos correspondientes
        data.loc[data['entrada'] == entry, 'salida'] = response
        data.loc[data['entrada'] == entry, 'fecha'] = fecha_hora_actual.split(',')[0]
        data.loc[data['entrada'] == entry, 'hora'] = fecha_hora_actual.split(',')[1]
        data.loc[data['entrada'] == entry, 'tipo_usuario'] = tipo_usuario
        data.loc[data['entrada'] == entry, 'nombre_usuario'] = nombre_usuario

        # Mostrar un mensaje indicando la actualización
        print(f"La entrada '{entry}' ha sido actualizada con la respuesta '{response}'.")

    return data

#entrenamiento del modelo
def train_model(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['entrada'])
    y = data['tipo']
 
    clf = MultinomialNB()
    clf.fit(X, y)
 
    return vectorizer, clf, data
#
def save_rnn_weights(model, weights_file):
    # Guardar únicamente los pesos del modelo
    model.save_weights(weights_file)

def load_rnn_weights(model, weights_file):
    # Cargar los pesos en el modelo
    model.load_weights(weights_file)
    
def save_rnn_model(model, tokenizer, label_encoder, weights_file, tokenizer_file, label_encoder_file):
    # Guardar el modelo completo (arquitectura y pesos)
    model.save(weights_file)
    model.save(model_file)
    # Guardar el tokenizer en formato JSON
    with open(tokenizer_file, 'w') as file:
        tokenizer_json = tokenizer.to_json()
        file.write(tokenizer_json)

    # Guardar el label encoder con joblib
    joblib.dump(label_encoder, label_encoder_file)


def load_rnn_model(model_file, tokenizer_file, label_encoder_file):
    # Cargar el modelo Keras
    model = load_model(model_file)

    # Cargar el tokenizer desde el archivo JSON
    with open(tokenizer_file, 'r') as file:
        tokenizer_json = file.read()
        tokenizer = tokenizer_from_json(tokenizer_json)

    # Cargar el label encoder con joblib
    label_encoder = joblib.load(label_encoder_file)

    return model, tokenizer, label_encoder

def train_rnn_model(input_data, output_data, max_sequence_length=4):
    global tokenizer, X

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(input_data)

    global X
    X = tokenizer.texts_to_sequences(input_data)
    X = pad_sequences(X, maxlen=max_sequence_length)
    MAX_SEQUENCE_LENGTH = X.shape[1]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(output_data)

    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=X.shape[1]))
    model.add(LSTM(150))
    
    model.add(Dropout(0.5))
    model.add(Dense(len(set(output_data)), activation='softmax'))
 
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, epochs=250, batch_size=46)
    return tokenizer, label_encoder, model

def save_data(data, filename):
    data.to_csv(filename, index=False)


def preprocess_rnn(text):
    # Convertir a minúsculas y eliminar caracteres no alfabéticos y números
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

    # Tokenización
    tokens = nltk.word_tokenize(text)

    # Lematización
    tokens = [lemmatizer_en.lemmatize(token) for token in tokens if token not in stop_words_en]

    return ' '.join(tokens)

def obtener_tipo_usuario():
    global configuracion_file

    configuracion_data = {'tipo': [], 'nombre': []}

    # Intentar cargar la configuración existente
    configuracion = load_data(configuracion_file)

    while True:
        tipo_usuario = input("¿Eres un entrenador o un usuario? (entrenador/usuario): ").lower()
        if tipo_usuario in ['entrenador', 'usuario']:
            if tipo_usuario == 'entrenador':
                nombre_usuario = input("Ingresa tu nombre de usuario (entrenador): ")
                contraseña = input("Ingresa tu contraseña: ")

                # Validar nombre de usuario y contraseña con el archivo de entrenadores
                entrenadores_df = pd.read_csv(entrenadores, encoding='latin1')
                if (not entrenadores_df.empty and
                    (entrenadores_df['usuario'] == nombre_usuario) & (entrenadores_df['contraseña'] == contraseña)).any():
                    print(f"Acceso autorizado como {tipo_usuario}.")
                    configuracion_data['tipo'].append(tipo_usuario)
                    configuracion_data['nombre'].append(nombre_usuario)
                    configuracion = pd.concat([configuracion, pd.DataFrame(configuracion_data)], ignore_index=True)
                    save_data(configuracion, configuracion_file)  # Guardar configuración en el archivo
                    return tipo_usuario
                else:
                    print("Nombre de usuario o contraseña incorrectos. Intenta de nuevo.")
            else:
                # Si es un usuario regular, simplemente almacenar el tipo de usuario y el nombre
                nombre_usuario = input("Ingresa tu nombre de usuario (usuario regular): ")
                configuracion_data['tipo'].append(tipo_usuario)
                configuracion_data['nombre'].append(nombre_usuario)
                configuracion = pd.concat([configuracion, pd.DataFrame(configuracion_data)], ignore_index=True)
                save_data(configuracion, configuracion_file)  # Guardar configuración en el archivo
                return tipo_usuario
        else:
            print("Por favor, elige 'entrenador' o 'usuario'.")
            
def get_response_tfidf(user_input, vectorizer, clf, data, data_file, nombre_usuario, configuracion):
    if not user_input.strip():
        print("Chatbot: Por favor, proporciona una entrada válida.")
        return None, None, None

    data = load_data(data_file)  # Cargar el conjunto de datos actualizado
    user_entry = preprocess_ingles(user_input)  # Guardar la entrada del usuario en una variable
    existing_entries = data['entrada'].apply(preprocess_ingles)

    # Calcular la similitud de coseno entre la entrada del usuario y las entradas existentes
    tfidf_matrix = vectorizer.transform(existing_entries)
    user_vector = vectorizer.transform([user_entry])
    similarities = cosine_similarity(tfidf_matrix, user_vector)

    # Obtener el índice de la entrada existente más similar
    most_similar_index = np.argmax(similarities)
    max_similarity_entry = similarities[most_similar_index]

    # Establecer un umbral para determinar si la similitud es lo suficientemente alta para la entrada
    entry_similarity_threshold = 0.85

    if max_similarity_entry < entry_similarity_threshold:
        print("No tengo la respuesta, puedes ayudarme para mi aprendizaje. similitud baja({:.2f}%). ".format(float(max_similarity_entry[0]) * 100))

        # Asegurarse de que configuracion no está vacío y contiene la columna 'tipo'
        if not configuracion.empty and 'tipo' in configuracion.columns:
            # Obtener el tipo de usuario solo si la columna 'tipo' existe
            tipo_usuario = configuracion['tipo'].iloc[0]

            if tipo_usuario == 'entrenador':
                new_response = input("Nueva respuesta: ")
                data = add_new_entry(data, user_input, new_response, nombre_usuario, configuracion)
                save_data(data, data_file)
                # Reentrenar el modelo después de agregar una nueva respuesta
                vectorizer, clf, data = train_model(data)
                return new_response, user_entry, ''
            else:
                print("Chatbot: No tengo la respuesta y no puedo aprender, ya que no eres un entrenador.")
                return None, None, None
        else:
            print("Chatbot: No se puede determinar el tipo de usuario.")
            return None, None, None

    # Calcular la similitud de coseno entre la respuesta del usuario y las respuestas existentes
    # Calcular la similitud de coseno entre la respuesta del usuario y las respuestas existentes
    response_vector = vectorizer.transform([user_entry])
    response_similarities = cosine_similarity(tfidf_matrix, response_vector)

    # Obtener el índice de la respuesta existente más similar
    most_similar_response_index = np.argmax(response_similarities)
    max_similarity_response = response_similarities[most_similar_response_index]

    # Establecer un umbral para determinar si la similitud es lo suficientemente alta para la respuesta
    response_similarity_threshold = 0.78

    if max_similarity_response < response_similarity_threshold:
        print(f"La similitud para la respuesta es baja ({max_similarity_response.item() * 100:.2f}%). Proporcione una nueva respuesta.")

        # Asegurarse de que configuracion no está vacío y contiene la columna 'tipo'
        if not configuracion.empty and 'tipo' in configuracion.columns:
            # Obtener el tipo de usuario solo si la columna 'tipo' existe
            tipo_usuario = configuracion['tipo'].iloc[0]

            if tipo_usuario == 'entrenador':
                new_response = input("Nueva respuesta: ")
                if new_response !="":
                    data = add_new_entry(data, user_input, new_response, nombre_usuario, configuracion)
                    save_data(data, data_file)
                    return new_response, user_entry, ''
                else:
                    new_response="Entiendo que no deseas dar informacion"
                    return '', '', ''
            else:
                print("Chatbot: La similitud para la respuesta es baja, y no puedo aprender ya que no eres un entrenador.")
                return None, None, None
        else:
            print("Chatbot: No se puede determinar el tipo de usuario.")
            return None, None, None

    existing_response = data.loc[most_similar_index, 'salida']
    chatbot_entry = existing_entries[most_similar_index]
    print("Chatbot:", existing_response)

    # No preguntar si la respuesta es correcta en el modo de usuario
    if configuracion['tipo'].iloc[0] == 'entrenador':
        is_correct = input("¿Es esta información correcta? (sí/no): ")

        if is_correct.lower() == 'no':
            updated_response = input("Por favor, proporciona la respuesta correcta (o deja en blanco para no proporcionar): ")

            if updated_response.strip() != '':
                # Comprobar si la entrada del usuario coincide con la entrada recuperada por el chatbot
                if user_entry != chatbot_entry:
                    # Agregar una nueva entrada en lugar de actualizar la existente
                    data = add_new_entry(data, user_input, updated_response, nombre_usuario, configuracion)
                else:
                    # Actualizar la respuesta existente
                    #data = update_entry(data, user_input, updated_response, nombre_usuario, configuracion)
                    data = update_entry(data, user_input, updated_response, configuracion)
                save_data(data, data_file)
                return updated_response, user_entry, chatbot_entry

    # Respuesta correcta, no actualizar user_entry y chatbot_entry si no se solicita la pregunta
    if '?' not in user_entry:
        user_entry = ''
        chatbot_entry = ''

   # print("Porcentaje de similitud (entrada): {:.2f}%".format(float(max_similarity_entry[0]) * 100))
   # print("Porcentaje de similitud (respuesta): {:.2f}%".format(float(max_similarity_response[0]) * 100))

    return existing_response, user_entry, chatbot_entry


def get_response(user_input, vectorizer, clf, tokenizer, rnn_model, label_encoder, data, data_file, nombre_usuario, configuracion):
    # Obtener respuestas de ambos modelos
    response_tfidf_nb = get_response_tfidf(user_input, vectorizer, clf, data, data_file, nombre_usuario, configuracion)
    response_rnn, _, _, _ = get_response_rnn(user_input, tokenizer, rnn_model, label_encoder, data, data_file, nombre_usuario, configuracion)
   


def get_response_options(response_tfidf_nb, response_rnn):
    # Imprimir respuestas
    print("Chatbot (TF-IDF):", response_tfidf_nb)
    print("Chatbot (RNN):", response_rnn)

    # Preguntar al usuario cuál respuesta prefiere
    preferred_response = input("¿Cuál respuesta prefieres? (tf-idf/rnn): ").lower()

    if preferred_response == 'tf-idf':
        return response_tfidf_nb
    elif preferred_response == 'rnn':
        return response_rnn
    else:
        return "Chatbot: Opción no válida. Por favor, elige entre 'tf-idf' y 'rnn'."


def get_response_rnn(user_input, tokenizer, rnn_model, label_encoder, data, data_file, nombre_usuario, configuracion):
    try:
        # Verificar si la entrada del usuario está vacía
        if not user_input.strip():
            print("Chatbot: Por favor, proporciona una entrada válida.")
            return None, None, None, None

        # Cargar el conjunto de datos actualizado
        data = load_data(data_file)

        # Preprocesar la entrada del usuario
        user_entry = preprocess_rnn(user_input)

        # Convertir la entrada del usuario a secuencia y realizar el padding
        user_sequence = tokenizer.texts_to_sequences([user_entry])
        
        # Asegurarse de que la longitud de la secuencia coincida con la esperada por el modelo
        user_padded = pad_sequences(user_sequence, maxlen=MAX_SEQUENCE_LENGTH)

        # Realizar la predicción con el modelo RNN
        prediction = rnn_model.predict(user_padded)

        # Decodificar la predicción para obtener la etiqueta
        predicted_label = np.argmax(prediction)

        # Obtener la respuesta asociada a la etiqueta predicha
        response_label = label_encoder.classes_[predicted_label]

        # Buscar la respuesta en el conjunto de datos
        response_data = data[data['salida'] == response_label]

        if response_data.empty:
            print("No tengo la respuesta. ¿Desea que el modelo vuelva a entrenarse?")
            retrain = input("¿Desea que el modelo vuelva a entrenarse? (s/n): ").lower()

            if retrain == 's':
                # Retraining del modelo
                tokenizer, label_encoder, rnn_model = train_rnn_model(data['entrada'], data['salida'], max_sequence_length=4)

                # Guardar el modelo después de entrenarlo
                save_rnn_model(rnn_model, tokenizer, label_encoder, weights_file, tokenizer_file, label_encoder_file)
                rnn_model, tokenizer, label_encoder = load_rnn_model(model_file, tokenizer_file, label_encoder_file)
                print("Modelo RNN ha sido reentrenado.")

                # Hacer una nueva predicción después de reentrenar
                prediction = rnn_model.predict(user_padded)
                predicted_label = np.argmax(prediction)
                response_label = label_encoder.classes_[predicted_label]

                response_data = data[data['salida'] == response_label]

                if response_data.empty:
                    print("Chatbot (RNN): A pesar de reentrenar el modelo, aún no tengo una respuesta para su entrada.")
                    return None, None, None, None

            else:
                print("Chatbot (RNN): Mil disculpas, no se obtuvo respuesta exacta para tu pregunta")
                return None, None, None, None

        response = response_data['salida'].iloc[0]

        # Imprimir la respuesta
        print("Chatbot (RNN):", response)

        # No preguntar si la respuesta fue útil o precisa en el modo de usuario
        if configuracion['tipo'].iloc[0] == 'entrenador':
            feedback = input("¿Fue útil o precisa la respuesta de la red neuronal? (sí/no): ").lower()

            if feedback == 'no':
                # Retraining del modelo
                tokenizer, label_encoder, rnn_model = train_rnn_model(data['entrada'], data['salida'], max_sequence_length=4)

                # Guardar el modelo después de entrenarlo
                save_rnn_model(rnn_model, tokenizer, label_encoder, weights_file, tokenizer_file, label_encoder_file)
                rnn_model, tokenizer, label_encoder = load_rnn_model(model_file, tokenizer_file, label_encoder_file)

    except Exception as e:
        print(f"Error en la función get_response_rnn: {e}")
        return None, None, None, None

    # Si no hay retroalimentación negativa o la respuesta del usuario es "sí", simplemente devolver la respuesta original
    return response, user_entry, user_input, response_label
def cambiar_usuario():
    global configuracion_file

    # Borrar la configuración existente
    save_data(pd.DataFrame(columns=['tipo', 'nombre']), configuracion_file)
    print("Se ha cambiado el usuario. Por favor, configura tu nuevo tipo de usuario.")
    return obtener_tipo_usuario()

# ... (código anterior)

def main():
    # Comprobar si los archivos del modelo y preprocesamiento existen
    if os.path.exists(weights_file) and os.path.exists(tokenizer_file) and os.path.exists(label_encoder_file):
         # Cargar modelo RNN previamente entrenado
        rnn_model, tokenizer, label_encoder = load_rnn_model(model_file, tokenizer_file, label_encoder_file)

    else:
        # Entrenar un nuevo modelo RNN si no se encuentra uno guardado
        data = load_data(data_file)
        tokenizer, label_encoder, rnn_model = train_rnn_model(data['entrada'], data['salida'], max_sequence_length=4)

        # Guardar el modelo RNN
        save_rnn_model(rnn_model, tokenizer, label_encoder, weights_file, tokenizer_file, label_encoder_file)

    try:
        vectorizer, clf, data = train_model(data)  # Actualiza la asignación
    except UnboundLocalError:
        # Manejar la excepción aquí, por ejemplo, cargando nuevamente los datos
        data = load_data(data_file)
        vectorizer, clf, data = train_model(data)  # Actualiza la asignación

    configuracion = load_data(configuracion_file)

    nombre_usuario = None  # Mueve la declaración aquí

    if configuracion.empty:
        print("¡Bienvenido! Antes de comenzar, configura tu tipo de usuario.")
        tipo_usuario = obtener_tipo_usuario()
    else:
        tipo_usuario = configuracion['tipo'].iloc[0]
        nombre_usuario = configuracion['nombre'].iloc[0] if 'nombre' in configuracion.columns else None

        if tipo_usuario == 'entrenador' and nombre_usuario:
            print(f"Bienvenido de nuevo, {nombre_usuario}!")
        elif tipo_usuario == 'usuario':
            print(f"¡Bienvenido, {nombre_usuario}!")
        else:
            print("¡Bienvenido!")

    print(f"\n Hola {nombre_usuario}, Inicia la conversación con el Chatbot:")
    while True:
        user_input = input("Usuario: ")
        if user_input.lower() == 'salir':
            confirmacion = input("¿Seguro que quieres salir y borrar la configuración? (s/n): ").lower()
            if confirmacion == 's':
                # Borrar la configuración existente
                save_data(pd.DataFrame(columns=['tipo', 'nombre']), configuracion_file)
                print("Configuración borrada. ¡Hasta luego!")
                break
            else:
                print("No se ha borrado la configuración. Continuando...")
        else:
            # Obtener y mostrar la respuesta
            response = get_response(user_input, vectorizer, clf, tokenizer, rnn_model, label_encoder, data, data_file, nombre_usuario, configuracion)
            
if __name__ == "__main__":
    main()
  
