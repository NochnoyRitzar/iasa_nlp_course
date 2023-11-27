from fastapi import FastAPI, Request
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.pipelines import pipeline as onnx_pipeline

model_path = "./Lecture_8/model/t5-small/q_int8"

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = ORTModelForSeq2SeqLM.from_pretrained(model_path, use_cache=False)
summarizer = onnx_pipeline('summarization', model=model, tokenizer=tokenizer)


@app.post("/summarize")
async def create_summary(request: Request):
    data = await request.body()
    data_str = data.decode("utf-8")

    summary = summarizer(data_str)
    return {"summary": summary[0]['summary_text']}