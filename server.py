import sentenceCompleter
import fastapi

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


@app.post("/tell_me_stories")
async def tell_me_stories(request: fastapi.Request):

  # text is in text field?
  text = (await request.json())["text"]
  result = sentenceCompleter.generate(text)
  print("text", text)
  print("result", result)
  return result
