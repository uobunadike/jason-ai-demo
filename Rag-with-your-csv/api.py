# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# from model import run

# app = FastAPI(title="Inventory Analytics API")

# # Allow Angular (localhost:4200 or deployed frontend)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change to frontend URL in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class QueryRequest(BaseModel):
#     query: str
#     model_type: str = "ollama"     # Optional
#     model_name: str = "llama3.1"   # Optional

# @app.post("/query")
# def query_inventory(data: QueryRequest):
#     try:
#         result = run(data.query, data.model_type, data.model_name)
#         return {"response": result}
#     except Exception as e:
#         return {"error": str(e)}



# from fastapi import FastAPI, HTTPException, Depends
# from pydantic import BaseModel
# from model import run
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI(
#     title="Inventory Analytics API",
#     version="1.0.0",
#     docs_url="/api/docs",
#     redoc_url="/api/redoc"
# )

# # CORS Configuration for Angular frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class AnalyticsRequest(BaseModel):
#     query: str
#     model_type: str = "ollama"
#     model_name: str = "llama3.1"

# class AnalyticsResponse(BaseModel):
#     result: str
#     model_used: str

# @app.post("/api/analyze", response_model=AnalyticsResponse)
# async def analyze_query(request: AnalyticsRequest):
#     try:
#         result = run(
#             query=request.query,
#             model_type=request.model_type,
#             model_name=request.model_name
#         )
#         return AnalyticsResponse(
#             result=result,
#             model_used=f"{request.model_type}/{request.model_name}"
#         )
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Analysis failed: {str(e)}"
#         )


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import run

app = FastAPI(
    title="Inventory Analytics API",
    version="1.0.0"
)

# Allow CORS for all origins (for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your Angular app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    model_type: str = "ollama"
    model_name: str = "llama3.1"

class QueryResponse(BaseModel):
    result: str

@app.post("/api/analyze", response_model=QueryResponse)
async def analyze(request: QueryRequest):
    try:
        result = run(request.query, request.model_type, request.model_name)
        return QueryResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
