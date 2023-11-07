import fastapi
import torch
# import tokenizer


app = fastapi.FastAPI()

# run with - uvicorn server:app --host 0.0.0.0 --port 8080 --reload
@app.on_event("startup")
async def startup_event():
  print("Starting Up")

  # app.state.tokenizer = tokenizer.Tokenizer()
  # app.state.tokenizer.embeddings.eval()
  
  # app.state.model.load_state_dict(torch.load("./model?????.pt"))
  # app.state.model.eval()



@app.on_event("shutdown")
async def startup_event():
  print("Shutting down")


@app.get("/")
def on_root():
  return { "message": "Hello App" }


@app.post("/learn_to_search")
async def learn_to_search(request: fastapi.Request):

  # query is in text field?
  query = (await request.json())["text"]

  # query_tokens = app.state.tokenizer.encode(query)
  # query_embeddings = app.state.tokenizer.embeddings(query_tokens)

  # document_indexes = app.state.model(query_embedddings)

  # return extended passage
  result = ["Mary had a little lamb"]
  print("query", query)
  print("result", result)
  return result
