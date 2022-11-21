import keras
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from flask import Flask, request, render_template, flash
app = Flask(__name__)

def paia_generator(seed_text):
    
    paia = seed_text
    next_words = 35

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([paia])[0]
        token_list = pad_sequences([token_list], maxlen=23, padding='pre')
        predict_x=modelo_payas.predict(token_list, verbose=0) 
        classes_x=np.argmax(predict_x,axis=1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == classes_x:
                output_word = word
                break
        paia += " " + output_word
    
    paia_list = paia.split()
    output = ""

    for i in range(len(paia_list)):
        if i==0:
            output+=paia_list[i]
        else:
            if i%6 == 0:
                output += "\n" + paia_list[i]
            else:
                output+= " " + paia_list[i]
    
    return output


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        palabra = request.form['palabra']

        if not palabra:
            processed_pal = ""
        else:
            processed_pal = paia_generator(palabra)

    else:
        processed_pal = None

    return render_template('index.html', processed_pal = processed_pal)
 
if __name__ == '__main__':
    # carga el modelo desde el archivo
    modelo_payas = keras.models.load_model('modelo_payas.h5')

    # carga el tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    app.run(debug=True)