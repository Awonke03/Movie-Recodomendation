
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
url = 'https://raw.githubusercontent.com/your-username/your-repo-name/main/movielens_100k.csv'

# Read the CSV file into a pandas DataFrame
dt = pd.read_csv(url)

dt.head()

dt.columns

dt.info()
len(dt.directors.unique())

len(dt.title.unique())

y=dt["title"]
X=dt.drop(columns="title")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import dash
from dash import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel



# Preprocess data for recommendation engine
dt['year'] = dt['year'].fillna('')
dt['directors'] = dt['directors'].fillna('')
dt['actors'] = dt['actors'].fillna('')
dt['genres'] = dt['genres'].fillna('')
dt['combined_features'] = dt['year'].astype(str) + ' ' + dt['directors'] + ' ' + dt['actors'] + ' ' + dt['genres']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(dt['combined_features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


fig1 = px.scatter(dt.explode('genres'), x='directors', y='genres', title="Relationship between Directors and Genres")
fig1.update_layout(
    xaxis_title="Director",
    yaxis_title="Genre",
    height=800,
    width=1350,
)

# Visualization 2: Number of Movies in Each Genre
fig2 = px.bar(dt['genres'].str.split(expand=True).stack().reset_index(name='Genre'), x='Genre', title="Number of Movies in Each Genre")
fig2.update_layout(xaxis_title="Genre", yaxis_title="Number of Movies", height=800, width=1350)

# Visualization 3: Top Directors with the Most Movies
fig3 = px.bar(dt['directors'].value_counts().reset_index(), x='index', y='directors', title="Top Directors with the Most Movies")
fig3.update_layout(xaxis_title="Director", yaxis_title="Number of Movies", height=800, width=1350)

years_count = dt.year.value_counts().sort_index()
df = pd.DataFrame({'Year': years_count.index, 'Count': years_count.values})
# Visualization 4: Yearly Counts
fig4 = px.line(df, x='Year', y='Count', markers=True, title='Yearly Counts',
              labels={'Count': 'Number of Occurrences', 'Year': 'Year'})
fig4.update_layout(height=800, width=1350)

# Create Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

dt.dropna(inplace=True)

# Visualization 5: Sunburst chart
fig5 = px.sunburst(dt, path=['year', 'genres', 'title'])
fig5.update_layout(title_text="Genres and Movies by Year", height=800, width=1350)

# Default layout
default_layout = html.Div([
    html.H1("Movie Recommendations", style={'text-align': 'center'}),

    dcc.Input(id='movie-input', type='text', placeholder='Enter a movie title', style={'width': '50%', 'margin': '10px'}),
    html.Button(id='submit-button', n_clicks=0, children='Get Recommendations', style={'margin': '10px'}),

    html.Div(id='recommendations-output', style={'margin': '20px'})
])

# Define app layout
app.layout = html.Div(children=[
    html.Div(
        className='menu',
        children=[
            html.Button('Search and Recommendations', id='btn-0', n_clicks=0, style={'backgroundColor': 'darkgrey', 'border-radius': '12px', 'margin-bottom': '10px'}),
            html.Button('Directors and Genres', id='btn-1', n_clicks=0, style={'backgroundColor': 'darkgrey', 'border-radius': '12px', 'margin-bottom': '10px'}),
            html.Button('Number of Movies in Each Genre', id='btn-2', n_clicks=0, style={'backgroundColor': 'darkgrey', 'border-radius': '12px', 'margin-bottom': '10px'}),
            html.Button('Top Directors with the Most Movies', id='btn-3', n_clicks=0, style={'backgroundColor': 'darkgrey', 'border-radius': '12px', 'margin-bottom': '10px'}),
            html.Button('Yearly Counts', id='btn-4', n_clicks=0, style={'backgroundColor': 'darkgrey', 'border-radius': '12px', 'margin-bottom': '10px'}),
            html.Button('Sunburst Chart', id='btn-5', n_clicks=0, style={'backgroundColor': 'darkgrey', 'border-radius': '12px', 'margin-bottom': '10px'}),
        ]
    ),

    html.Div(id='page-content', style={'backgroundColor': 'darkgrey', 'padding': '20px'}),
])

# Callback to update page content based on button clicks
@app.callback(Output('page-content', 'children'),
              [Input('btn-0', 'n_clicks'),
               Input('btn-1', 'n_clicks'),
               Input('btn-2', 'n_clicks'),
               Input('btn-3', 'n_clicks'),
               Input('btn-4', 'n_clicks'),
               Input('btn-5', 'n_clicks')])
def display_page(btn0, btn1, btn2, btn3, btn4, btn5):
    ctx = dash.callback_context
    button_id = ctx.triggered_id.split('.')[0] if ctx.triggered_id else 'btn-0'

    if button_id == 'btn-0':
        return default_layout
    elif button_id == 'btn-1':
        return dcc.Graph(figure=fig1)
    elif button_id == 'btn-2':
        return dcc.Graph(figure=fig2)
    elif button_id == 'btn-3':
        return dcc.Graph(figure=fig3)
    elif button_id == 'btn-4':
        return dcc.Graph(figure=fig4)
    elif button_id == 'btn-5':
        return dcc.Graph(figure=fig5)
    else:
        return default_layout

# Callback to handle button click and display recommendations
@app.callback(Output('recommendations-output', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('movie-input', 'value')])
def get_recommendations(n_clicks, movie_title):
    if n_clicks > 0 and movie_title:
        if movie_title not in dt['title'].values:
            return f"Movie with title '{movie_title}' not found in the dataset."

        idx = dt[dt['title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        recommended_movies = dt['title'].iloc[movie_indices]

        return html.Ul([html.Li(movie) for movie in recommended_movies])

    return ""

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050)
