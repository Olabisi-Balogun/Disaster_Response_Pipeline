import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import plotly.graph_objs as obj
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    #Visualization of messages based on common disaster
    disaster_cat = ['floods','storm','fire','earthquake']
    df_dis = df[['message','floods','storm','fire','earthquake']]
    flood_count = len(df_dis[df_dis['floods']==1])
    storm_count = len(df_dis[df_dis['storm']==1])
    fire_count = len(df_dis[df_dis['fire']==1])
    earthquake_count = len(df_dis[df_dis['earthquake']==1])

    #Visualize Basic needs
    basic_need = ['water','food','shelter','clothing']
    basic_need_df = df[['message','water','food','clothing','shelter']]
    water_count = len(basic_need_df[basic_need_df['water']==1])
    food_count = len(basic_need_df[basic_need_df['food']==1])
    clothing_count = len(basic_need_df[basic_need_df['clothing']==1])
    shelter_count = len(basic_need_df[basic_need_df['shelter']==1])


    
    # create visuals
    bar = [obj.Bar(
            x = genre_names,
            y = genre_counts
            )]

        
    graph1 = obj.Figure(bar)
    graph1.update_layout(
        title_text = 'Distribution of Message Genres',
        title_font_size = 24,
        xaxis=dict(title='Genre'),
        yaxis = dict(title = 'Count'))

    colors = ['#83d7ee','#5d6066','#e25822','#3b3e42']

    graph2 = obj.Figure(
        data = [obj.Bar(x = disaster_cat,
        y = [flood_count, storm_count, fire_count, earthquake_count],
        marker_color=colors)],
        layout = dict(title=dict(text="Message Classification by Disasters"),
        xaxis = dict(title = 'Disasters'),
        yaxis = dict(title='Count')))

    graph3 = obj.Figure(
        data = [obj.Bar(x = basic_need,
            y = [water_count, food_count, clothing_count, shelter_count])],
        layout = dict(title = dict(text="Message Classification By Needed Amenities"),
            xaxis = dict(title = 'Basic Amenities'),
            yaxis = dict(title = 'Count')))




    
    # encode plotly graphs in JSON
    graphJSON1 = json.dumps(graph1, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON2 = json.dumps(graph2, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON3 = json.dumps(graph3, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    #return render_template('master.html', ids=ids, graphJSON=graphJSON)
    return render_template('master.html', plot1=graphJSON1, plot2 =graphJSON2,
        plot3 = graphJSON3)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()