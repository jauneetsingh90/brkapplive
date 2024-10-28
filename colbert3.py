import os
import time
import openai
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_loading_spinners as dls
from dotenv import load_dotenv
import torch
from ragstack_langchain.colbert import ColbertVectorStore as LangchainColbertVectorStore
from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel
from util.config import LOGGER, ASTRA_DB_ID, ASTRA_TOKEN

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Set torch device to CPU to avoid CUDA issues
device = torch.device("cpu")
torch.set_default_tensor_type("torch.FloatTensor")

# Initialize the embedding model and database connection
embedding_model = ColbertEmbeddingModel()
database = CassandraDatabase.from_astra(
    astra_token=ASTRA_TOKEN,
    database_id=ASTRA_DB_ID,
    keyspace='default_keyspace'
)

# Initialize the vector store
lc_vector_store = LangchainColbertVectorStore(
    database=database,
    embedding_model=embedding_model,
)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("ColBERT Similarity Search with GPT-4 Enhancement", className="text-center"),
            dbc.Input(
                id="question-input",
                type="text",
                placeholder="Ask your question here",
                className="mb-2"
            ),
            dbc.Button("Search", id="submit-button", color="primary", className="mb-4")
        ])
    ], className="justify-content-center"),

    dbc.Row([
        # Raw Similarity Search Results Section
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Raw Similarity Search Results"),
                dbc.CardBody([
                    dls.Hash(
                        id="similarity-spinner",
                        children=[
                            dcc.Markdown(id="similarity-result", className="result-content"),
                            html.Div(id="similarity-time", className="result-time"),
                            html.Div(id="similarity-usage-metadata", className="usage-metadata")
                        ],
                        color="#7BD1F5"
                    )
                ])
            ], className="mb-4")
        ], width=12),
    ]),

    dbc.Row([
        # GPT-4 Enhanced Results Section
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("GPT-4 Enhanced Summary of Results"),
                dbc.CardBody([
                    dls.Hash(
                        id="gpt-enhanced-result-spinner",
                        children=[
                            dcc.Markdown(id="gpt-enhanced-result", className="result-content"),
                            html.Div(id="gpt-enhanced-time", className="result-time"),
                            html.Div(id="gpt-enhanced-usage-metadata", className="usage-metadata")
                        ],
                        color="#7BD1F5"
                    )
                ])
            ], className="mb-4")
        ], width=12),
    ])
], fluid=True)

# Define the callback for updating both raw and enhanced similarity results
@app.callback(
    [Output("similarity-result", "children"),
     Output("similarity-time", "children"),
     Output("similarity-usage-metadata", "children"),
     Output("gpt-enhanced-result", "children"),
     Output("gpt-enhanced-time", "children"),
     Output("gpt-enhanced-usage-metadata", "children")],
    [Input("submit-button", "n_clicks")],
    [State("question-input", "value")]
)
def update_results(n_clicks, question):
    if n_clicks > 0 and question:
        try:
            start_time = time.time()
            # Execute similarity search
            docs = lc_vector_store.similarity_search(query=question, k=5)
            elapsed_time = time.time() - start_time

            # Prepare the raw results text
            raw_result_text = "\n\n".join([f"{doc.metadata['source']} - {doc.page_content[:150]}..." for doc in docs])

            # GPT-4 prompt to enhance the response
            gpt_prompt = f"The following are the search results for the query '{question}':\n\n{raw_result_text}\n\n" \
                         "Please summarize these results in a user-friendly and informative way."

            # Call GPT-4 to refine the result text
            gpt_start_time = time.time()
            gpt_response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes search results."},
                    {"role": "user", "content": gpt_prompt}
                ]
            )
            gpt_elapsed_time = time.time() - gpt_start_time

            # Extract the refined response
            refined_response = gpt_response.choices[0].message.content

            return (raw_result_text,
                    f"Elapsed time: {elapsed_time:.2f} seconds", f"Found {len(docs)} documents",
                    refined_response,
                    f"GPT-4 processing time: {gpt_elapsed_time:.2f} seconds", "GPT-4 used for enhanced summary")
        
        except Exception as e:
            print(f"Error in similarity search or GPT-4 enhancement: {e}")
            return "An error occurred during similarity search.", "", "", "An error occurred with GPT-4 enhancement.", "", ""
    return "", "", "", "", "", ""

if __name__ == "__main__":
    app.run_server(debug=True, port=8052, host="0.0.0.0")