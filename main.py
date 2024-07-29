import streamlit as st
import pickle
import nltk
nltk.download('stopwords')


# Load the model and vectorizer
with open('fake_news_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.sav', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define the preprocessing function
def stemming(content):
    import re
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    port_stem = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [port_stem.stem(word) for word in content if word not in stop_words]
    content = ' '.join(content)
    return content

# Streamlit UI
st.title('Fake News Detection')
st.write('Enter the text of the news article below:')

user_input = st.text_area("News Article Text")

if st.button('Predict'):
    if user_input:
        # Pre-process the text
        processed_text = stemming(user_input)
        
        # Convert the text to numerical data
        text_vector = vectorizer.transform([processed_text])
        
        # Make the prediction
        prediction = model.predict(text_vector)
        result = 'Real' if prediction[0] == 0 else 'Fake'
        
        st.write(f'The news is: {result}')
    else:
        st.write('Please enter some text to classify.')
