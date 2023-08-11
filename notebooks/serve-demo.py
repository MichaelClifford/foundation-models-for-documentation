import ray 
from ray import serve
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus":1, "num_gpus":0})
@serve.ingress(app)
class servedGen:
    def __init__(self):
        self._model = pipeline("text2text-generation", model="google/flan-t5-small")
    @app.post("/") 
    def generate(self, text:str) -> str:
        return self._model(text)[0]["generated_text"]
    
text_app = servedGen.bind()